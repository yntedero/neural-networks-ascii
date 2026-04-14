# Uloha 2 - ASCII Art
# Neuronove siete na rozpoznavanie znakov a konverzia obrazkov na ASCII text.
#
# Autor: Ostapchuk

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

# seed - aby vysledky boli vzdy rovnake
torch.manual_seed(42)
np.random.seed(42)

print("PyTorch verzia:", torch.__version__)

# znaky, ktore rozpoznavame
# medzera je kodovana ako nuly (prazdny blok)
ZNAKY = [" ", "|", "/", "\\", "_", "-", "^", "o"]
POCET_ZNAKOV = len(ZNAKY)
SIRKA_BLOKU = 8
VYSKA_BLOKU = 14
VECTORSIZE = SIRKA_BLOKU * VYSKA_BLOKU


# ============================================================
# Priprava datasetu
# ============================================================

# [AI] programove generovanie binarnych sablon znakov (8x14 pixelov)
# namiesto manualneho zadavania poli pouzivame funkcie na kreslenie tvarov
# kazdy znak ma niekolko variant pre lepsiu generalizaciu

def _empty():
    return np.zeros((VYSKA_BLOKU, SIRKA_BLOKU), dtype=np.float32)


def _fill_row(img, row, x1, x2, val=1.0):
    img[row, max(0, x1):min(SIRKA_BLOKU, x2)] = val


def _fill_col(img, col, y1, y2, val=1.0):
    img[max(0, y1):min(VYSKA_BLOKU, y2), col] = val


def _fill_diag(img, direction, thickness=1, step=2):
    for y in range(VYSKA_BLOKU):
        col = y // step
        if direction == "/":
            x = SIRKA_BLOKU - 1 - col
        else:
            x = col
        for t in range(thickness):
            px = x + t
            if 0 <= px < SIRKA_BLOKU:
                img[y, px] = 1.0


def _fill_circle(img, cy, cx, radius, filled=True):
    for y in range(VYSKA_BLOKU):
        for x in range(SIRKA_BLOKU):
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if filled:
                if dist <= radius:
                    img[y, x] = 1.0
            else:
                if abs(dist - radius) < 0.8:
                    img[y, x] = 1.0


def _fill_triangle(img, top_y, top_x, base_width, height, val=1.0):
    for row in range(height):
        if top_y + row >= VYSKA_BLOKU:
            break
        half = (row + 1) * base_width / (2 * height)
        x1 = int(top_x - half)
        x2 = int(top_x + half + 0.5)
        _fill_row(img, top_y + row, x1, x2, val)


# kazdy znak ma niekolko geometrickych variant pre lepsiu generalizaciu
def vytvor_znakove_sablony():
    sablony = {}

    sablony[" "] = [_empty()]

    sablony["|"] = []
    for col in [4, 3, 5]:
        t = _empty()
        _fill_col(t, col, 0, VYSKA_BLOKU)
        sablony["|"].append(t)
    for col in [3]:
        t = _empty()
        _fill_col(t, col, 0, VYSKA_BLOKU)
        _fill_col(t, col + 1, 0, VYSKA_BLOKU)
        sablony["|"].append(t)

    sablony["/"] = []
    for thick in [1, 2]:
        t = _empty()
        _fill_diag(t, "/", thickness=thick)
        sablony["/"].append(t)

    sablony["\\"] = []
    for thick in [1, 2]:
        t = _empty()
        _fill_diag(t, "\\", thickness=thick)
        sablony["\\"].append(t)

    sablony["_"] = []
    for row in [13, 12]:
        t = _empty()
        _fill_row(t, row, 0, SIRKA_BLOKU)
        sablony["_"].append(t)

    sablony["-"] = []
    for row in [6, 5, 7]:
        t = _empty()
        _fill_row(t, row, 1, SIRKA_BLOKU - 1)
        sablony["-"].append(t)
    for row in [5]:
        t = _empty()
        _fill_row(t, row, 1, SIRKA_BLOKU - 1)
        _fill_row(t, row + 1, 1, SIRKA_BLOKU - 1)
        sablony["-"].append(t)

    sablony["^"] = []
    t = _empty()
    _fill_triangle(t, 0, 4, 8, 4)
    sablony["^"].append(t)
    t = _empty()
    _fill_row(t, 0, 4, 5)
    _fill_row(t, 1, 3, 6)
    _fill_row(t, 2, 2, 7)
    _fill_row(t, 3, 1, 8)
    sablony["^"].append(t)
    t = _empty()
    _fill_triangle(t, 0, 4, 8, 4)
    _fill_row(t, 1, 3, 6)
    sablony["^"].append(t)

    sablony["o"] = []
    for cy, r in [(7, 2.5), (6, 2.5), (6, 2.2)]:
        t = _empty()
        _fill_circle(t, cy, 3.5, r, filled=True)
        sablony["o"].append(t)
    t = _empty()
    _fill_circle(t, 6, 3.5, 2.5, filled=False)
    sablony["o"].append(t)

    return sablony


# [AI] augmentacne funkcie (posun, sum) som navrhol s pomocou AI
# augmentacia zvacsuje dataset umelymi variantmi

def posun_obrazka(img, dx, dy):
    vysledok = np.zeros_like(img)
    for y in range(VYSKA_BLOKU):
        for x in range(SIRKA_BLOKU):
            ny, nx = y + dy, x + dx
            if 0 <= ny < VYSKA_BLOKU and 0 <= nx < SIRKA_BLOKU:
                vysledok[ny, nx] = img[y, x]
    return vysledok


def pridaj_sum(img, intensita=0.1):
    sum = np.random.uniform(-intensita, intensita, img.shape).astype(np.float32)
    return np.clip(img + sum, 0.0, 1.0)


# [AI] rotacia obrazka pomocou PIL - pre augmentaciu datasetu
def rotacia_obrazka(img, uhol):
    pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
    rotated = pil_img.rotate(uhol, resample=Image.BICUBIC, expand=False, fillcolor=0)
    return np.array(rotated, dtype=np.float32) / 255.0


# dataset obsahuje posunutia, sum a rotacie - realne obrazky su vzdy trochu ine
def vytvor_dataset(sablony, augmentacia=True):
    vstupy = []
    vystupy = []

    for idx, znak in enumerate(ZNAKY):
        one_hot = np.zeros(POCET_ZNAKOV, dtype=np.float32)
        one_hot[idx] = 1.0

        for sablona in sablony[znak]:
            flat = sablona.flatten()
            vstupy.append(flat)
            vystupy.append(one_hot)

            if augmentacia and znak != " ":
                for dx in [-2, -1, 1, 2]:
                    for dy in [-1, 1]:
                        posunuty = posun_obrazka(sablona, dx, dy)
                        vstupy.append(posunuty.flatten())
                        vystupy.append(one_hot)

                for _ in range(2):
                    zasumeny = pridaj_sum(sablona, 0.1)
                    vstupy.append(zasumeny.flatten())
                    vystupy.append(one_hot)

                for uhol in [-10, -5, 5, 10]:
                    otoceny = rotacia_obrazka(sablona, uhol)
                    vstupy.append(otoceny.flatten())
                    vystupy.append(one_hot)

    X = torch.tensor(np.array(vstupy), dtype=torch.float32)
    Y = torch.tensor(np.array(vystupy), dtype=torch.float32)
    return X, Y


sablony = vytvor_znakove_sablony()
print("Znaky:", ZNAKY)
print("Pocet sablon:", sum(len(v) for v in sablony.values()))

X_all, Y_all = vytvor_dataset(sablony, augmentacia=True)
dataset = TensorDataset(X_all, Y_all)
print(f"Dataset: {len(dataset)} vzoriek (s augmentaciou)")


# ============================================================
# Architektury neuronovych sieti
# ============================================================

# Model 1: mensia siet
# 2 skryte vrstvy, malo parametrov

class NetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(VECTORSIZE, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, POCET_ZNAKOV)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Model 2: stredna siet
# 3 skryte vrstvy, stredny pocet parametrov

class NetMedium(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(VECTORSIZE, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, POCET_ZNAKOV)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


# Model 3: vacsia siet
# [AI] inspiracia pre hlbku siete z AI - vacsia kapacita pre zlozitejsie vzory
# 4 skryte vrstvy s ReLU, dropout pre regularizaciu

class NetLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(VECTORSIZE, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, POCET_ZNAKOV)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x


print("Modely pripravene")
print(f"  NetSmall:  {sum(p.numel() for p in NetSmall().parameters())} parametrov")
print(f"  NetMedium: {sum(p.numel() for p in NetMedium().parameters())} parametrov")
print(f"  NetLarge:  {sum(p.numel() for p in NetLarge().parameters())} parametrov")


# ============================================================
# Trenovanie a testovanie (mini-batch)
# ============================================================

# trenovanie siete s mini-batch
# pouzivame SSE chybu a SGD optimizer
# [AI] prechod na mini-batch trenovanie pre rychlejsi beh

BATCH_SIZE = 32


def train(model, X, Y, epochs, lr, print_every=500):
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    historia = []

    for epoch in range(1, epochs + 1):
        celkova_chyba = 0.0

        for batch_X, batch_Y in loader:
            vystup = model(batch_X)
            chyba = ((vystup - batch_Y) ** 2).mean()

            optimizer.zero_grad()
            chyba.backward()
            optimizer.step()

            celkova_chyba += chyba.item()

        historia.append(celkova_chyba)

        if epoch % print_every == 0 or epoch == epochs:
            print(f"Epocha {epoch}  Global error  {celkova_chyba:.5f}")

    return historia


# testovanie na trenovacich datach
# accuracy = kolko znakov bolo rozpoznanych spravne
# reliability = kolko vystupov je blizko 0 alebo 1

def test(model, X, Y, epsilon=0.2):
    model.eval()
    n = X.shape[0]
    spravne = 0
    celk_acc = 0
    celk_rel = 0

    print("\nTestovanie")
    print(f"{'Znak':<6}{'Predikcia':<10}{'Output':<30}{'Error':<10}{'Accuracy':<12}{'Reliability'}")
    print("-" * 100)

    with torch.no_grad():
        for i in range(n):
            vstup = X[i]
            ciel = Y[i]
            vystup = model(vstup)
            pred_idx = torch.argmax(vystup).item()
            true_idx = torch.argmax(ciel).item()
            chyba = torch.sum((vystup - ciel) ** 2).item()

            acc = (torch.round(vystup) == ciel).sum().item() / len(ciel) * 100
            celk_acc += acc

            rel_ok = ((vystup < epsilon) | (vystup > 1 - epsilon)).sum().item()
            rel = rel_ok / len(vystup) * 100
            celk_rel += rel

            if pred_idx == true_idx:
                spravne += 1

            pred_znak = ZNAKY[pred_idx]
            true_znak = ZNAKY[true_idx]
            out_str = " ".join([f"{v:.2f}" for v in vystup])
            print(f"{true_znak!r:<6}{pred_znak!r:<10}{out_str:<30}{chyba:<10.3f}{acc:.0f}%{'':<8}{rel:.0f}%")

    print(f"\nSpravne: {spravne}/{n}")
    print(f"Priemerna accuracy: {celk_acc / n:.1f}%")
    print(f"Priemerna reliability: {celk_rel / n:.1f}%")

    model.train()
    return spravne, celk_acc / n, celk_rel / n


# ============================================================
# Konverzia obrazka na ASCII
# ============================================================

# rozrezanie obrazka na bloky a konverzia na ASCII
# [AI] logiku rozrezania obrazka som navrhol s pomocou AI

def obrazok_na_ascii(model, cesta_obrazka, sirka_bloku=8, vyska_bloku=14):
    img = Image.open(cesta_obrazka).convert("L")
    img_array = np.array(img, dtype=np.float32) / 255.0

    vyska, sirka = img_array.shape
    bloky_x = sirka // sirka_bloku
    bloky_y = vyska // vyska_bloku

    bloky = []
    for by in range(bloky_y):
        for bx in range(bloky_x):
            y1 = by * vyska_bloku
            y2 = y1 + vyska_bloku
            x1 = bx * sirka_bloku
            x2 = x1 + sirka_bloku
            blok = img_array[y1:y2, x1:x2].flatten()
            bloky.append(blok)

    all_blocks = torch.tensor(np.array(bloky), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        vystupy = model(all_blocks)

    ascii_vysledok = []
    idx = 0
    for by in range(bloky_y):
        riadok = ""
        for bx in range(bloky_x):
            pred = torch.argmax(vystupy[idx]).item()
            riadok += ZNAKY[pred]
            idx += 1
        ascii_vysledok.append(riadok)

    model.train()
    return "\n".join(ascii_vysledok)


# ============================================================
# Experimenty
# ============================================================

# ============================================================
# Model 1: NetSmall (112->64->32->8)
# ============================================================

print("\n" + "=" * 60)
print("MODEL 1: NetSmall (112->64->32->8)")
print("=" * 60)

# Experiment 1.1 - zakladny, lr=1.0, 500 epoch (mini-batch = menej epoch)
print("\n=== Experiment 1.1 (lr=1.0, 500 epoch) ===\n")

torch.manual_seed(42)
np.random.seed(42)
model_1_1 = NetSmall()
hist_1_1 = train(model_1_1, X_all, Y_all, epochs=500, lr=1.0, print_every=100)
s_1_1, a_1_1, r_1_1 = test(model_1_1, X_all, Y_all)

# Experiment 1.2 - vyssie lr=2.0
print("\n=== Experiment 1.2 (lr=2.0, 800 epoch) ===\n")

torch.manual_seed(42)
np.random.seed(42)
model_1_2 = NetSmall()
hist_1_2 = train(model_1_2, X_all, Y_all, epochs=800, lr=2.0, print_every=200)
s_1_2, a_1_2, r_1_2 = test(model_1_2, X_all, Y_all)

# Experiment 1.3 - krokovy learning rate
# [AI] napad krokoveho lr som nasiel s pomocou AI
print("\n=== Experiment 1.3 (krokovy lr) ===\n")

torch.manual_seed(42)
np.random.seed(42)
model_1_3 = NetSmall()
hist_1_3 = []
print("--- faza 1: lr=2.0, 300 epoch ---")
h = train(model_1_3, X_all, Y_all, epochs=300, lr=2.0, print_every=150)
hist_1_3.extend(h)
print("\n--- faza 2: lr=0.5, 200 epoch ---")
h = train(model_1_3, X_all, Y_all, epochs=200, lr=0.5, print_every=100)
hist_1_3.extend(h)
print("\n--- faza 3: lr=0.05, 100 epoch ---")
h = train(model_1_3, X_all, Y_all, epochs=100, lr=0.05, print_every=50)
hist_1_3.extend(h)
s_1_3, a_1_3, r_1_3 = test(model_1_3, X_all, Y_all)

# graf chyb - model 1
plt.figure(figsize=(10, 4))
plt.plot(hist_1_1, label="Exp 1.1 (lr=1.0)")
plt.plot(hist_1_2, label="Exp 1.2 (lr=2.0)")
plt.plot(hist_1_3, label="Exp 1.3 (krokovy lr)")
plt.xlabel("Epocha")
plt.ylabel("Global error")
plt.title("Model 1 (NetSmall) - Priebeh chyby")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

n_data = len(dataset)
print(f"Exp 1.1 (lr=1.0):    spravne={s_1_1}/{n_data}  acc={a_1_1:.1f}%  rel={r_1_1:.1f}%")
print(f"Exp 1.2 (lr=2.0):    spravne={s_1_2}/{n_data}  acc={a_1_2:.1f}%  rel={r_1_2:.1f}%")
print(f"Exp 1.3 (step lr):   spravne={s_1_3}/{n_data}  acc={a_1_3:.1f}%  rel={r_1_3:.1f}%")


# ============================================================
# Model 2: NetMedium (112->128->64->32->8)
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2: NetMedium (112->128->64->32->8)")
print("=" * 60)

# Experiment 2.1 - zakladny, lr=1.0
print("\n=== Experiment 2.1 (lr=1.0, 500 epoch) ===\n")

torch.manual_seed(42)
np.random.seed(42)
model_2_1 = NetMedium()
hist_2_1 = train(model_2_1, X_all, Y_all, epochs=500, lr=1.0, print_every=100)
s_2_1, a_2_1, r_2_1 = test(model_2_1, X_all, Y_all)

# Experiment 2.2 - vyssie lr=2.0
print("\n=== Experiment 2.2 (lr=2.0, 800 epoch) ===\n")

torch.manual_seed(42)
np.random.seed(42)
model_2_2 = NetMedium()
hist_2_2 = train(model_2_2, X_all, Y_all, epochs=800, lr=2.0, print_every=200)
s_2_2, a_2_2, r_2_2 = test(model_2_2, X_all, Y_all)

# Experiment 2.3 - krokovy learning rate
# [AI] napad krokoveho lr som nasiel s pomocou AI
print("\n=== Experiment 2.3 (krokovy lr) ===\n")

torch.manual_seed(42)
np.random.seed(42)
model_2_3 = NetMedium()
hist_2_3 = []
print("--- faza 1: lr=2.0, 300 epoch ---")
h = train(model_2_3, X_all, Y_all, epochs=300, lr=2.0, print_every=150)
hist_2_3.extend(h)
print("\n--- faza 2: lr=0.5, 200 epoch ---")
h = train(model_2_3, X_all, Y_all, epochs=200, lr=0.5, print_every=100)
hist_2_3.extend(h)
print("\n--- faza 3: lr=0.05, 100 epoch ---")
h = train(model_2_3, X_all, Y_all, epochs=100, lr=0.05, print_every=50)
hist_2_3.extend(h)
s_2_3, a_2_3, r_2_3 = test(model_2_3, X_all, Y_all)

# graf chyb - model 2
plt.figure(figsize=(10, 4))
plt.plot(hist_2_1, label="Exp 2.1 (lr=1.0)")
plt.plot(hist_2_2, label="Exp 2.2 (lr=2.0)")
plt.plot(hist_2_3, label="Exp 2.3 (krokovy lr)")
plt.xlabel("Epocha")
plt.ylabel("Global error")
plt.title("Model 2 (NetMedium) - Priebeh chyby")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# Model 3: NetLarge (112->256->128->64->32->8)
# ============================================================

print("\n" + "=" * 60)
print("MODEL 3: NetLarge (112->256->128->64->32->8)")
print("=" * 60)

# Experiment 3.1 - zakladny, lr=0.5
print("\n=== Experiment 3.1 (lr=0.5, 800 epoch) ===\n")

torch.manual_seed(42)
np.random.seed(42)
model_3_1 = NetLarge()
hist_3_1 = train(model_3_1, X_all, Y_all, epochs=800, lr=0.5, print_every=200)
s_3_1, a_3_1, r_3_1 = test(model_3_1, X_all, Y_all)

# Experiment 3.2 - vyssie lr=2.0
print("\n=== Experiment 3.2 (lr=2.0, 800 epoch) ===\n")

torch.manual_seed(42)
np.random.seed(42)
model_3_2 = NetLarge()
hist_3_2 = train(model_3_2, X_all, Y_all, epochs=800, lr=2.0, print_every=200)
s_3_2, a_3_2, r_3_2 = test(model_3_2, X_all, Y_all)

# Experiment 3.3 - krokovy learning rate
# [AI] napad krokoveho lr som nasiel s pomocou AI
print("\n=== Experiment 3.3 (krokovy lr) ===\n")

torch.manual_seed(42)
np.random.seed(42)
model_3_3 = NetLarge()
hist_3_3 = []
print("--- faza 1: lr=2.0, 300 epoch ---")
h = train(model_3_3, X_all, Y_all, epochs=300, lr=2.0, print_every=150)
hist_3_3.extend(h)
print("\n--- faza 2: lr=0.5, 200 epoch ---")
h = train(model_3_3, X_all, Y_all, epochs=200, lr=0.5, print_every=100)
hist_3_3.extend(h)
print("\n--- faza 3: lr=0.05, 100 epoch ---")
h = train(model_3_3, X_all, Y_all, epochs=100, lr=0.05, print_every=50)
hist_3_3.extend(h)
s_3_3, a_3_3, r_3_3 = test(model_3_3, X_all, Y_all)

# graf chyb - model 3
plt.figure(figsize=(10, 4))
plt.plot(hist_3_1, label="Exp 3.1 (lr=0.5)")
plt.plot(hist_3_2, label="Exp 3.2 (lr=2.0)")
plt.plot(hist_3_3, label="Exp 3.3 (krokovy lr)")
plt.xlabel("Epocha")
plt.ylabel("Global error")
plt.title("Model 3 (NetLarge) - Priebeh chyby")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# Ulozenie modelov
# ============================================================

os.makedirs("models", exist_ok=True)

torch.save(model_1_1.state_dict(), "models/model1_exp1.pth")
torch.save(model_1_2.state_dict(), "models/model1_exp2.pth")
torch.save(model_1_3.state_dict(), "models/model1_exp3.pth")
torch.save(model_2_1.state_dict(), "models/model2_exp1.pth")
torch.save(model_2_2.state_dict(), "models/model2_exp2.pth")
torch.save(model_2_3.state_dict(), "models/model2_exp3.pth")
torch.save(model_3_1.state_dict(), "models/model3_exp1.pth")
torch.save(model_3_2.state_dict(), "models/model3_exp2.pth")
torch.save(model_3_3.state_dict(), "models/model3_exp3.pth")
print("\nModely ulozene do priecinka models/")


# ============================================================
# Test na realnych obrazkoch
# ============================================================

print("\n" + "=" * 60)
print("TEST NA REALNYCH OBRAZKOCH")
print("=" * 60)

cesta = "test-photo.png"

best_models = [
    ("NetSmall (exp 1.3)", model_1_3),
    ("NetMedium (exp 2.3)", model_2_3),
    ("NetLarge (exp 3.3)", model_3_3),
]

if os.path.exists(cesta):
    print(f"\n{'=' * 50}")
    print(f"OBRAZOK: {cesta}")
    print(f"{'=' * 50}")
    for meno, model in best_models:
        print(f"\n--- {meno} ---")
        ascii_art = obrazok_na_ascii(model, cesta, SIRKA_BLOKU, VYSKA_BLOKU)
        print(ascii_art)
else:
    print(f"\nObrazok {cesta} nenajdeny")


# ============================================================
# Zhrnutie
#
# Porovnali sme 3 architektury:
# - NetSmall: 2 skryte vrstvy (64, 32) s Sigmoid
# - NetMedium: 3 skryte vrstvy (128, 64, 32) s Sigmoid
# - NetLarge: 4 skryte vrstvy (256, 128, 64, 32) s ReLU
#
# Optimalizacie:
# - Mini-batch trenovanie (batch_size=32) namiesto sample-by-sample SGD
# - TensorDataset + DataLoader pre efektivne batchovanie
# - Znizeny pocet epoch vdaka lepsiemu gradientu z mini-batchov
# - matplotlib Agg backend (zabrana blokovanie plt.show())
# - Vektorizovana konverzia obrazku na ASCII (vsetky bloky naraz)
#
# Co som zistil:
# - Vacsi model = lepsia schopnost generalizacie
# - Krokovy lr je opat najlepsia strategia
# - ReLU v hlbsej sieti pomaha rychlejsiemu uceniu
# - Pri realnom obrazku najlepsi vysledok dava NetLarge
