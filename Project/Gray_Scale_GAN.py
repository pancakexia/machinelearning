import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torchvision import models
import os
import csv
import numpy as np
from skimage import color
from skimage.color import lab2rgb
from skimage.metrics import structural_similarity as ssim

# 1. Transformations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

# 2. Datasets
train_dataset = STL10(root='./data', split='train', download=True, transform=transform_train)
val_dataset = STL10(root='./data', split='test', download=True, transform=transform_val)

# 3. Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
val_iter = iter(val_loader)

# 4. Attention Block
def attention_block(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels // 8, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels // 8, in_channels, 1),
        nn.Sigmoid()
    )

# 5. U-Net Generator with Attention
def down_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

def up_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=2):
        super(UNetGenerator, self).__init__()

        self.down1 = down_block(input_nc, 64)
        self.attn1 = attention_block(64)
        self.down2 = down_block(64, 128)
        self.attn2 = attention_block(128)
        self.down3 = down_block(128, 256)
        self.attn3 = attention_block(256)
        self.down4 = down_block(256, 512)
        self.attn4 = attention_block(512)

        self.up1 = up_block(512, 256)
        self.up2 = up_block(512, 128)
        self.up3 = up_block(256, 64)

        self.final = nn.Conv2d(128, output_nc, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d1 = self.attn1(d1) * d1
        d2 = self.down2(d1)
        d2 = self.attn2(d2) * d2
        d3 = self.down3(d2)
        d3 = self.attn3(d3) * d3
        d4 = self.down4(d3)
        d4 = self.attn4(d4) * d4

        u1 = self.up1(d4)
        u1 = torch.cat([u1, d3], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)

        out = torch.tanh(self.final(u3))
        return out

# 6. Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            down_block(input_nc, 64),
            down_block(64, 128),
            down_block(128, 256),
            down_block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)

# 7. VGG16 for Perceptual Loss
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
for param in vgg.parameters():
    param.requires_grad = False
vgg = vgg.cuda()

# 8. Loss Functions
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

def perceptual_loss(fake_rgb, real_rgb):
    fake_features = vgg(fake_rgb)
    real_features = vgg(real_rgb)
    return F.l1_loss(fake_features, real_features)

# 9. Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = UNetGenerator().to(device)
discriminator = Discriminator().to(device)

# 10. Optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# 11. Labels for GAN
real_label = 0.9
fake_label = 0.0

# 12. Helper to convert RGB to L and ab channels
def rgb_to_lab(img):
    img = img.permute(1, 2, 0).cpu().numpy()
    lab = color.rgb2lab(img).astype("float32")
    lab = torch.from_numpy(lab).permute(2, 0, 1)
    L = lab[:1, :, :] / 50.0 - 1.0
    ab = lab[1:, :, :] / 110.0
    return L, ab

def lab_to_rgb(L, ab):
    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    lab = torch.cat([L, ab], dim=0).cpu().numpy()
    lab = np.transpose(lab, (1, 2, 0))
    rgb = lab2rgb(lab)
    return (rgb * 255).astype(np.uint8)

# 13. Output directories
os.makedirs('output_images', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Lists to store losses and SSIM scores
d_losses = []
g_losses = []
ssim_scores = []

# 14. Training Loop
num_epochs = 200
for epoch in range(num_epochs):
    for batch_idx, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)

        L_channels = []
        ab_channels = []
        for img in imgs:
            L, ab = rgb_to_lab(img)
            L_channels.append(L)
            ab_channels.append(ab)

        L = torch.stack(L_channels).to(device)
        ab = torch.stack(ab_channels).to(device)

        # Train Discriminator
        discriminator.zero_grad()
        real_pair = torch.cat([L, ab], dim=1)
        pred_real = discriminator(F.interpolate(real_pair, size=(48, 48), mode='bilinear', align_corners=False))
        real_targets = torch.full_like(pred_real, real_label, device=device)
        d_loss_real = adversarial_loss(pred_real, real_targets)

        fake_ab = generator(L)
        L_down = F.interpolate(L, size=(fake_ab.size(2), fake_ab.size(3)), mode='bilinear', align_corners=False)
        fake_pair = torch.cat([L_down, fake_ab.detach()], dim=1)
        pred_fake = discriminator(fake_pair)
        fake_targets = torch.full_like(pred_fake, fake_label, device=device)
        d_loss_fake = adversarial_loss(pred_fake, fake_targets)

        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        generator.zero_grad()
        fake_ab = generator(L)
        fake_pair = torch.cat([L_down, fake_ab], dim=1)
        pred_fake = discriminator(fake_pair)
        real_targets = torch.full_like(pred_fake, real_label, device=device)
        g_loss_adv = adversarial_loss(pred_fake, real_targets)

        ab_down = F.interpolate(ab, size=(fake_ab.size(2), fake_ab.size(3)), mode='bilinear', align_corners=False)
        g_loss_l1 = l1_loss(fake_ab, ab_down)

        # Perceptual loss
        L_resized = F.interpolate(L, size=(fake_ab.size(2), fake_ab.size(3)), mode='bilinear', align_corners=False)
        fake_rgb = torch.cat([(L_resized + 1.0) * 0.5, (fake_ab * 0.5) + 0.5], dim=1)
        real_rgb = torch.cat([(L_resized + 1.0) * 0.5, (ab_down * 0.5) + 0.5], dim=1)
        g_loss_perc = perceptual_loss(fake_rgb, real_rgb)

        g_loss = g_loss_adv + 50 * g_loss_l1 + 0.1 * g_loss_perc

        g_loss.backward()
        g_optimizer.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            generator.eval()
            with torch.no_grad():
                try:
                    val_imgs, _ = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_imgs, _ = next(val_iter)

                val_imgs = val_imgs.to(device)
                n_samples = 5
                sample_imgs = val_imgs[:n_samples]
                L_channels = []
                for img in sample_imgs:
                    L, _ = rgb_to_lab(img)
                    L_channels.append(L)
                sample_L = torch.stack(L_channels).to(device)

                sample_fake_ab = generator(sample_L)
                sample_fake_ab_upsampled = F.interpolate(sample_fake_ab, size=(96, 96), mode='bilinear', align_corners=False)

                imgs_out = []
                ssim_batch = []
                for i in range(n_samples):
                    orig_img = (sample_imgs[i].cpu() * 255).byte()
                    gray_img = lab_to_rgb(sample_L[i], torch.zeros_like(sample_L[i].repeat(2, 1, 1)))
                    fake_rgb = lab_to_rgb(sample_L[i], sample_fake_ab_upsampled[i])

                    real_img_np = np.transpose(orig_img.numpy(), (1, 2, 0))
                    fake_img_np = fake_rgb
                    score = ssim(real_img_np, fake_img_np, channel_axis=2)
                    ssim_batch.append(score)

                    imgs_out.append(orig_img)
                    imgs_out.append(torch.from_numpy(gray_img).permute(2, 0, 1))
                    imgs_out.append(torch.from_numpy(fake_rgb).permute(2, 0, 1))

                avg_ssim = np.mean(ssim_batch)
                ssim_scores.append(avg_ssim)
                print(f"SSIM Score: {avg_ssim:.4f}")

                grid = vutils.make_grid(imgs_out, nrow=3)
                vutils.save_image(grid.float() / 255.0, f"output_images/val_epoch_{epoch+1}_batch_{batch_idx}.png")
            generator.train()

    torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch+1}.pth")
    torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch+1}.pth")

# Save Losses and SSIM to CSV
with open('losses.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['d_loss', 'g_loss'])
    for d, g in zip(d_losses, g_losses):
        writer.writerow([d, g])

with open('ssim_scores.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ssim_score'])
    for score in ssim_scores:
        writer.writerow([score])

