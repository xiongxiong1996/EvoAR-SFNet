import os
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR,CosineAnnealingWarmRestarts
from utils.data import DataSet
from models import EvoARFSNet
import torch.nn.functional as F
import inspect
import shutil
import h5py
from torch.utils.data import ConcatDataset
import numpy as np

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

class LossWeightEvolution:
    def __init__(self, population_size=10, max_generations=100, initial_mutation_rate=0.03, min_mutation_rate=0.001, top_k=3, seed=42):
        self.population_size = population_size
        self.initial_mutation_rate = initial_mutation_rate
        self.max_generations = max_generations
        self.min_mutation_rate = min_mutation_rate
        self.top_k = top_k
        self.current_generation = 0
        self.weights_dim = 7  # loss1, loss2, loss3, loss_f
        self.rng = np.random.default_rng(seed)
        self.population = self._initialize_population()

    def _initialize_population(self):
        pop = []
        base_weights = np.array([0.05, 0.05, 0.1, 0.05, 0.05, 0.1, 0.6])
        
        for _ in range(self.population_size):
            # Add small random perturbation to the base weights
            perturbation = 0.1 * (self.rng.random(self.weights_dim) - 0.5)  # Small random values around 0
            w = base_weights + perturbation
            
            # Ensure weights are positive and sum to 1
            w = np.clip(w, 0.01, None)  # Set minimum value to avoid zeros
            w /= w.sum()
            pop.append(w)
        return np.array(pop)

    def select_top(self, fitness_scores):
        top_indices = np.argsort(fitness_scores)[-self.top_k:]
        return self.population[top_indices]

    def crossover_and_mutate(self, top_individuals):
        # 线性退火：随着进化轮数增加，变异率逐渐降低
        progress = self.current_generation / self.max_generations
        self.mutation_rate = self.initial_mutation_rate * (1 - progress) + self.min_mutation_rate * progress
        new_population = []
        for _ in range(self.population_size):
            parents = self.rng.choice(len(top_individuals), 2, replace=False)
            alpha = self.rng.uniform(0, 1)
            child = alpha * top_individuals[parents[0]] + (1 - alpha) * top_individuals[parents[1]]
            # 使用当前动态调整的 mutation_rate
            mutation = self.rng.normal(0, self.mutation_rate, size=self.weights_dim)
            child += mutation
            child = np.clip(child, 1e-4, 1.0)
            child /= child.sum()
            new_population.append(child)
            
        self.population = np.array(new_population)
        self.current_generation += 1  # 更新进化轮数

    def next_generation(self, fitness_scores):
        top_individuals = self.select_top(fitness_scores)
        self.crossover_and_mutate(top_individuals)

    def get_population(self):
        return self.population
    

# 备份代码
def backup_code(checkpoint_save_path):
    # 构造 code 目录路径
    code_dir = os.path.join(checkpoint_save_path, 'code')
    os.makedirs(code_dir, exist_ok=True)

    # 获取当前运行的 trainer.py 文件路径
    current_script_path = os.path.abspath(inspect.stack()[1].filename)

    # 设置要备份的文件列表
    files_to_backup = [
        './models/models.py',
        current_script_path
    ]

    for file_path in files_to_backup:
        if os.path.exists(file_path):
            dst = os.path.join(code_dir, os.path.basename(file_path))
            shutil.copyfile(file_path, dst)
            print(f'Copied {file_path} to {dst}')
        else:
            print(f'Warning: {file_path} not found.')


def save_checkpoint(
    model, optimizer, scheduler, epoch, save_path
):  # save model function
    check_point = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    save_path = (
        save_path
        + "/"
        + f"checkpoint_{epoch}_"
        + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        + ".pth"
    )
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(check_point, save_path)

def save_bestpoint(
    model, optimizer, scheduler, epoch, save_path
):  # save model function
    check_point = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(check_point, save_path)


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs, lr, ckpt, batch_size, hw_range, task, checkpoint_save_path, fusion_type = (
        config.epochs, config.lr, config.ckpt, config.batch_size,
        config.hw_range, config.task, config.checkpoint_save_path, config.fusion_type
    )
    # 备份代码
    backup_code(checkpoint_save_path)
    train_set_path, checkpoint_path = config.train_set_path, config.checkpoint_path
    val_set_path = config.val_set_path
    train_set = DataSet(file_path=train_set_path)
    val_set = DataSet(file_path=val_set_path)
    # test_set_path = config.test_set_path
    # test_set = DataSet(file_path=test_set_path)
    # combined_dataset = ConcatDataset([train_set, val_set])
    test_data_loader= DataLoader(
        dataset=val_set,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    if task == "wv3":
        pan_channels, lms_channels = 1, 8
    elif task in ["qb", "gf2"]:
        pan_channels, lms_channels = 1, 4

    model = EvoARFSNet(pan_channels, lms_channels, fusion_type=fusion_type).to(device)

    # model = nn.DataParallel(model)
    criterion = nn.L1Loss().to(device)
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999)
    )
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=160, T_mult=1, eta_min=1e-6)
    epoch = 1
    final_weights = [0.05, 0.05, 0.1, 0.05, 0.05, 0.1, 0.6]
    freeze_conv_epoch = 100
    freeze_evo_epoch = 160
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"=> successfully loaded checkpoint from '{checkpoint_path}'")
    print("Start training...")
    evolver = LossWeightEvolution(population_size=3,max_generations=freeze_evo_epoch-freeze_conv_epoch, top_k=2)

    while epoch <= epochs + 1:
        current_population = evolver.get_population()
        fitness_scores = []
        if epoch <=freeze_evo_epoch:
            if epoch <=freeze_conv_epoch:
                    best_weights = [0.05, 0.05, 0.1, 0.05, 0.05, 0.1, 0.6]
                    weights = best_weights
                    model.train()
                    for iteration, batch in tqdm(enumerate(training_data_loader), total=len(training_data_loader), bar_format="{l_bar}{bar:10}{r_bar}"):
                        gt = batch[0].to(device)
                        lms = batch[1].to(device)
                        pan = batch[4].to(device)
                        optimizer.zero_grad()
                        outs1, outs2, outs3,outf1,outf2,outf3, out_fused = model(pan, lms, epoch, hw_range=hw_range)
                        loss1 = criterion(outs1, gt)
                        loss2 = criterion(outs2, gt)
                        loss3 = criterion(outs3, gt)
                        loss4 = criterion(outf1, gt)
                        loss5 = criterion(outf2, gt)
                        loss6 = criterion(outf3, gt)
                        loss_f = criterion(out_fused, gt)
                        # Evolutionary weight loss
                        loss = (weights[0] * loss1 +
                                weights[1] * loss2 +
                                weights[2] * loss3 +
                                weights[3] * loss4 +
                                weights[4] * loss5 +
                                weights[5] * loss6 +
                                weights[6] * loss_f)
                        loss.backward()
                        optimizer.step()
                                        
                    # === 验证阶段（关键修改） ===
                    model.eval()
                    total_l1 = 0.0
                    with torch.no_grad():
                        for test_batch in test_data_loader:
                            gt = test_batch[0].to(device)
                            lms = test_batch[1].to(device)
                            pan = test_batch[4].to(device)
                            _, _, _, _, _, _, out_fused = model(pan, lms, epoch, hw_range=hw_range)
                            total_l1 += F.l1_loss(out_fused, gt).item()  # 单个值
                    avg_l1 = total_l1 / len(test_data_loader)
                    fitness_scores.append(-avg_l1)  # 注意这里：要最大化 L1 的负值 == 最小化 L1
                    model.train()
                    print(f"Epoch {epoch} | Init Weights: {best_weights}| Fitness: {fitness_scores[0]:.6f}")
                    # 保存模型
                    if epoch % ckpt == 0 or (epoch - 1) % 100 == 0:
                        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_save_path)
            else:
                for weights in current_population:
                    model.train()
                    for iteration, batch in tqdm(enumerate(training_data_loader), total=len(training_data_loader), bar_format="{l_bar}{bar:10}{r_bar}"):
                        gt = batch[0].to(device)
                        lms = batch[1].to(device)
                        pan = batch[4].to(device)
                        optimizer.zero_grad()
                        outs1, outs2, outs3,outf1,outf2,outf3, out_fused = model(pan, lms, epoch, hw_range=hw_range)
                        loss1 = criterion(outs1, gt)
                        loss2 = criterion(outs2, gt)
                        loss3 = criterion(outs3, gt)
                        loss4 = criterion(outf1, gt)
                        loss5 = criterion(outf2, gt)
                        loss6 = criterion(outf3, gt)
                        loss_f = criterion(out_fused, gt)
                        # Evolutionary weight loss
                        loss = (weights[0] * loss1 +
                                weights[1] * loss2 +
                                weights[2] * loss3 +
                                weights[3] * loss4 +
                                weights[4] * loss5 +
                                weights[5] * loss6 +
                                weights[6] * loss_f)
                        loss.backward()
                        # 对当前权重组产生的梯度做归一化
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0/len(current_population))
                        optimizer.step()
                        
                    # === 验证阶段（关键修改） ===
                    model.eval()
                    total_l1 = 0.0
                    with torch.no_grad():
                        for test_batch in test_data_loader:
                            gt = test_batch[0].to(device)
                            lms = test_batch[1].to(device)
                            pan = test_batch[4].to(device)
                            _, _, _, _, _, _, out_fused = model(pan, lms, epoch, hw_range=hw_range)
                            total_l1 += F.l1_loss(out_fused, gt).item()  # 单个值
                    avg_l1 = total_l1 / len(test_data_loader)
                    fitness_scores.append(-avg_l1)  # 注意这里：要最大化 L1 的负值 == 最小化 L1
                    model.train()
                # 进化
                evolver.next_generation(fitness_scores)
                # 打印当前最优权重
                # 获取排序后的索引（从大到小）
                sorted_indices = np.argsort(fitness_scores)[::-1]
                # 取前两个索引
                top2_indices = sorted_indices[:2]

                # 获取对应的权重（关键修正）
                weight1 = current_population[top2_indices[0]]  # 第一个最佳权重
                weight2 = current_population[top2_indices[1]]  # 第二个最佳权重
                final_weights = [(w1+w2)/2 for w1,w2 in zip(weight1, weight2)]
                best_weights = current_population[top2_indices[0]]
                best_weights = best_weights.tolist()
                print(f"Epoch {epoch} | Best Weights: {best_weights} | Final Weights: {final_weights}| Fitness: {fitness_scores[top2_indices[0]]:.6f}")
                if epoch % ckpt == 0 or (epoch - 1) % 100 == 0:
                        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_save_path)
             
        else:
            best_weights = final_weights
            weights = best_weights
            model.train()
            for iteration, batch in tqdm(enumerate(training_data_loader), total=len(training_data_loader), bar_format="{l_bar}{bar:10}{r_bar}"):
                gt = batch[0].to(device)
                lms = batch[1].to(device)
                pan = batch[4].to(device)
                optimizer.zero_grad()
                outs1, outs2, outs3,outf1,outf2,outf3, out_fused = model(pan, lms, epoch, hw_range=hw_range)
                loss1 = criterion(outs1, gt)
                loss2 = criterion(outs2, gt)
                loss3 = criterion(outs3, gt)
                loss4 = criterion(outf1, gt)
                loss5 = criterion(outf2, gt)
                loss6 = criterion(outf3, gt)
                loss_f = criterion(out_fused, gt)
                # Evolutionary weight loss
                loss = (weights[0] * loss1 +
                        weights[1] * loss2 +
                        weights[2] * loss3 +
                        weights[3] * loss4 +
                        weights[4] * loss5 +
                        weights[5] * loss6 +
                        weights[6] * loss_f)
                loss.backward()
                optimizer.step()  
            # === 验证阶段（关键修改） ===
            model.eval()
            total_l1 = 0.0
            with torch.no_grad():
                        for test_batch in test_data_loader:
                            gt = test_batch[0].to(device)
                            lms = test_batch[1].to(device)
                            pan = test_batch[4].to(device)
                            _, _, _, _, _, _, out_fused = model(pan, lms, epoch, hw_range=hw_range)
                            total_l1 += F.l1_loss(out_fused, gt).item()  # 单个值
            avg_l1 = total_l1 / len(test_data_loader)
            fitness_scores.append(-avg_l1)  # 注意这里：要最大化 L1 的负值 == 最小化 L1
            model.train()
            print(f"Epoch {epoch} | Init Weights: {best_weights}| Fitness: {fitness_scores[0]:.6f}")
            # 保存模型
            if epoch % ckpt == 0 or (epoch - 1) % 100 == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_save_path)

        scheduler.step()
        # 日志
        os.makedirs(checkpoint_save_path, exist_ok=True)
        loss_file_path = os.path.join(checkpoint_save_path, "loss.txt")
        with open(loss_file_path, "a") as f:
            f.write(f"epoch: {epoch} | best_weight: {best_weights} | fitness: {fitness_scores[0]:.6f}\n")
        epoch += 1
def load_set(file_path):
    data = h5py.File(file_path)
    lms = torch.from_numpy(np.array(data['lms'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute([1, 0, 2, 3, 4])
    ms = torch.from_numpy(np.array(data['ms'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute([1, 0, 2, 3, 4])
    pan = torch.from_numpy(np.array(data['pan'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute([1, 0, 2, 3, 4])
    gt = torch.from_numpy(np.array(data['gt'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute([1, 0, 2, 3, 4])
    
    return lms.float(), ms.float(), pan.float(), gt.float()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size used in the training and validation loop.",
    )
    parser.add_argument(
        "--epochs", default=560, type=int, help="Total number of epochs."
    )
    parser.add_argument(
        "--lr",
        default=0.0006,
        type=float,
        help="Base learning rate at the start of the training.",
    )
    parser.add_argument(
        "--ckpt", default=10, type=int, help="Save model every ckpt epochs."
    )
    parser.add_argument(
        "--train_set_path", default="./pansharpening/training_data/train_wv3.h5", type=str, help="Path to the training set."
    )
    parser.add_argument(
        "--val_set_path", default="./pansharpening/validation_data/valid_wv3.h5", type=str, help="Path to the training set."
    )
    parser.add_argument(
        "--checkpoint_path", default="", type=str, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--checkpoint_save_path",
        default="./workdir/test",
        type=str,
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--hw_range",
        nargs=2,
        type=int,
        default=[0, 18],
        help="The range of the height and width.",
    )
    parser.add_argument("--use_pretrain", action="store_true", help="...")
    parser.add_argument(
        "--task",
        default="wv3",
        type=str,
        choices=["wv3", "qb", "gf2"],
        help="Model to train (choices: wv3, qb, gf2).",
    )
    parser.add_argument(
        "--fusion_type",
        default="implicit",
        type=str,
        choices=["add", "concat", "explicit", "implicit"],
        help="Fusion type for the spatial-frequency cross-domain fusion.",
    )
    config = parser.parse_args()
    train(config)
