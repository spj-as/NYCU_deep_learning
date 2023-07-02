import numpy as np
import os
import torch
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.optim import Adam
from model import Unet
import random
import argparse
from plot import plot
from dataset import iclevr_dataset
from evaluator import evaluation_model
from tqdm import tqdm


def add_noise(x, current_step, args):
    beta = (args.end_beta - args.start_beta) * torch.arange(
        0, args.t_steps + 1, dtype=torch.float32
    ) / args.t_steps + args.start_beta
    alpha = 1 - beta
    log_alpha = torch.log(alpha)
    cumulated_alpha = torch.cumsum(log_alpha, dim=0).exp()
    alpha_bar = torch.sqrt(cumulated_alpha).to(args.device)
    alpha_bar_sqrt = torch.sqrt(1 - cumulated_alpha).to(args.device)

    noise = torch.randn_like(x).to(args.device)

    x_with_noise = (
        alpha_bar[current_step, None, None, None] * x
        + alpha_bar_sqrt[current_step, None, None, None] * noise
    )
    return x_with_noise, noise


def reverse(x_t, noise, current_step, args):
    beta_range = (args.end_beta - args.start_beta) * torch.arange(
        0, args.t_steps + 1
    ).float() / args.t_steps + args.start_beta
    alpha = 1 - beta_range
    log_alpha = torch.log(alpha)
    cumulated_alpha = torch.cumsum(log_alpha, dim=0).exp()

    alpha_over_sqrt_one_minus_alphabar = (
        (1 - alpha) / torch.sqrt(1 - cumulated_alpha)
    ).to(args.device)
    one_over_alpha = 1 / torch.sqrt(alpha).to(args.device)
    beta_sqrt = torch.sqrt(beta_range).to(args.device)

    epsilon_coefficient = alpha_over_sqrt_one_minus_alphabar[
        current_step, None, None, None
    ]
    epsilon = torch.randn(x_t.shape, device=args.device)
    x_t_minus_one = (one_over_alpha[current_step, None, None, None]) * (
        x_t - epsilon_coefficient * noise
    ) + (beta_sqrt[current_step, None, None, None]) * epsilon
    return x_t_minus_one


def denoise_image(ddpm, test_data, args):
    ddpm.eval()
    pred_x = torch.randn(32, 3, 64, 64).to(args.device)
    for label in tqdm(test_data):
        image_x = denoise_with_steps(ddpm, pred_x, label, args)
    return image_x, label


def denoise_with_steps(ddpm, pred_x, label, args):
    current_x = pred_x.clone()
    for step in range(args.t_steps):
        step_time = torch.tensor([args.t_steps - step - 1] * 32).to(args.device)
        with torch.no_grad():
            pred_noise = ddpm(current_x, step_time, label.to(args.device).long())
            current_x = reverse(current_x, pred_noise, step_time, args)
    return current_x


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--start_beta", default=0.0001, type=float)
    parser.add_argument("--end_beta", default=0.01, type=float)
    parser.add_argument("--model_dir", default="./model/")
    parser.add_argument("--t_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=550)
    parser.add_argument("--seed", type=int, default=226)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()
    set_seed(226)
    train_dataset = iclevr_dataset("train.json", "train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_dataset = iclevr_dataset("test.json", "test")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    new_test_dataset = iclevr_dataset("new_test.json", "test")
    new_test_dataloader = torch.utils.data.DataLoader(
        new_test_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    ddpm = Unet().to(args.device)
    optimizer = Adam(ddpm.parameters(), lr=args.lr)
    loss_function = F.mse_loss
    train_loss = []
    test_acc = []
    best_acc = 0.0
    if not args.test_only:
        for epoch in tqdm(range(args.epochs)):
            ddpm.train()
            for x, label in tqdm(train_dataloader):
                optimizer.zero_grad()
                x = x.to(args.device)
                label = label.to(args.device)
                current_step = torch.randint(0, args.t_steps, (x.shape[0],)).long()
                x_noised, noise = add_noise(x, current_step, args)
                pred = ddpm(
                    x_noised.to(args.device),
                    current_step.to(args.device),
                    label.to(args.device).long(),
                )
                loss = F.mse_loss(noise.to(args.device), pred)
                loss.backward()
                optimizer.step()
            train_loss.append(loss.item())
            print(f"[epoch: %02d] loss: %.5f \n" % (epoch, loss.item()))
            pred_x, test_label = denoise_image(ddpm, test_dataloader, args)
            Accuracy = evaluation_model().eval(pred_x, test_label)
            test_acc.append(Accuracy)
            print(f"Test Accuracy: %.5f \n" % (Accuracy))
            if Accuracy > best_acc:
                torch.save(ddpm.state_dict(), "Unet.pt")
                best_acc = Accuracy
                print(
                    f"Save best test accuracy model! Best Accuracy: %.5f \n"
                    % (Accuracy)
                )
                image = make_grid(pred_x, nrow=8, normalize=True)
                save_image(image, "best_result.png")

        plot(args.epochs, train_loss, test_acc)

    ddpm.load_state_dict(torch.load("Unet.pt"))
    # test
    pred_x, test_label = denoise_image(ddpm, test_dataloader, args)
    Accuracy = evaluation_model().eval(pred_x, test_label)
    print(f"Test Accuracy: %.5f \n" % (Accuracy))
    path = os.path.join("./test/test_result.png")
    save_image(make_grid(pred_x, nrow=8, normalize=True), path)
    # new test
    pred_x, test_label = denoise_image(ddpm, new_test_dataloader, args)
    Accuracy = evaluation_model().eval(pred_x, test_label)
    print(f"New Test Accuracy: %.5f \n" % (Accuracy))
    path = os.path.join("./test/new_test_result.png")
    save_image(make_grid(pred_x, nrow=8, normalize=True), path)
