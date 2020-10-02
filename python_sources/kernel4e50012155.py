import torch
import torchvision
from torch.functional import F
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np

# Debug part, skip this in submitted variant ==========================================

# Feature flags
DEBUG_LOG = True
CLIP_DATASETS = False
TRAIN_MODEL = True
LOG_EVERY = 1


def log(message, depth=0, same_line=False):
    if DEBUG_LOG:
        from termcolor import colored
        from datetime import datetime

        if log.last_was_same_line:
            print()

        log.last_was_same_line = same_line

        message = "  " * depth + str(message)
        if same_line:
            print("\r", end="")
        print(colored(f"[{datetime.now()}]", "cyan"), end=" ")
        if same_line:
            print(colored(message, "white"), end="")
        else:
            print(colored(message, "white"))


log.last_was_same_line = False


class FunctionLogContext:
    depth = 0

    @classmethod
    def before_fn(cls, message):
        log(message, cls.depth)
        cls.depth += 1

    @classmethod
    def after_fn(cls):
        cls.depth -= 1


def fnlog(message):
    from functools import wraps

    def wrapper(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            FunctionLogContext.before_fn(message.format(*args, **kwargs))
            result = fn(*args, **kwargs)
            FunctionLogContext.after_fn()
            return result

        return wrapped_fn

    return wrapper


def dlog(message, same_line=False):
    log(message, FunctionLogContext.depth, same_line=same_line)


def init_same_line():
    log("")


# Read from here on in submitted variant ===============================================


@fnlog("Loading datasets")
def load_and_transform_datasets():
    mnist_train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        shuffle=True,
        batch_size=1024,
        drop_last=True,
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "./data",
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        shuffle=True,
        batch_size=1024,
        drop_last=True,
    )
    svhn_train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            "./data",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        shuffle=True,
        batch_size=1024,
        drop_last=True,
    )
    svhn_test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            "./data",
            split="test",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        shuffle=True,
        batch_size=1024,
        drop_last=True,
    )
    return mnist_train_loader, mnist_test_loader, svhn_train_loader, svhn_test_loader


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = torch.nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv1_2 = torch.nn.Conv2d(32, 32, (3, 3), padding=1)
        self.pool1 = torch.nn.MaxPool2d((2, 2))

        self.conv2_1 = torch.nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv2_2 = torch.nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv2_3 = torch.nn.Conv2d(64, 64, (3, 3), padding=1)
        self.pool2 = torch.nn.MaxPool2d((2, 2))

        self.conv3_1 = torch.nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv3_2 = torch.nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_3 = torch.nn.Conv2d(128, 128, (3, 3), padding=1)
        self.pool3 = torch.nn.MaxPool2d((2, 2))

        self.drop1 = torch.nn.Dropout()

        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = torch.nn.Linear(128, 10)

        torch.nn.init.xavier_normal_(self.conv1_1.weight)
        torch.nn.init.xavier_normal_(self.conv1_2.weight)
        torch.nn.init.xavier_normal_(self.conv2_1.weight)
        torch.nn.init.xavier_normal_(self.conv2_2.weight)
        torch.nn.init.xavier_normal_(self.conv2_3.weight)
        torch.nn.init.xavier_normal_(self.conv3_1.weight)
        torch.nn.init.xavier_normal_(self.conv3_2.weight)
        torch.nn.init.xavier_normal_(self.conv3_3.weight)
        torch.nn.init.xavier_normal_(self.fc4.weight)
        torch.nn.init.xavier_normal_(self.fc5.weight)

        torch.nn.utils.weight_norm(self.conv1_1, "weight")
        torch.nn.utils.weight_norm(self.conv1_2, "weight")
        torch.nn.utils.weight_norm(self.conv2_1, "weight")
        torch.nn.utils.weight_norm(self.conv2_2, "weight")
        torch.nn.utils.weight_norm(self.conv2_3, "weight")
        torch.nn.utils.weight_norm(self.conv3_1, "weight")
        torch.nn.utils.weight_norm(self.conv3_2, "weight")
        torch.nn.utils.weight_norm(self.conv3_3, "weight")
        torch.nn.utils.weight_norm(self.fc4, "weight")

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(F.relu(self.conv2_3(x)))

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(F.relu(self.conv3_3(x)))

        x_c = F.avg_pool2d(x, 4)
        x_c = x_c.view(-1, 128)
        x_c = self.drop1(x_c)

        x_c = F.relu(self.fc4(x_c))
        x_c = self.fc5(x_c)
        return x, x_c


class OldWeightEMA(object):
    """
    ref: https://github.com/Britefury/self-ensemble-visual-domain-adapt/blob/a9808a1377fbe4248f627d4e44d44b1dc27b1959/optim_weight_ema.py#L4
    """

    def __init__(self, target_net, source_net, alpha=0.9):
        self.target_params = list(target_net.parameters())
        self.source_params = list(source_net.parameters())
        self.alpha = alpha

        for p, src_p in zip(self.target_params, self.source_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.target_params, self.source_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)


class EnsembleDA:
    def __init__(self, device):
        self.device = device
        self.student = Net().to(device)
        self.teacher = Net().to(device)

        self.student_optimizer = torch.optim.Adam(self.student.parameters())
        self.teacher_optimizer = OldWeightEMA(self.teacher, self.student)

    @fnlog("Training model")
    def train(
        self, source_dataset, target_dataset, source_val_dataset, target_val_dataset
    ):
        EPOCHS = 30

        def unsup_loss_fn(student, teacher):
            conf_teacher = torch.max(teacher, 1)[0]
            unsup_mask = (conf_teacher > 0.96837722).float()

            aug_loss = (student - teacher) ** 2
            aug_loss = aug_loss.mean(dim=1)

            unsup_loss = (aug_loss * unsup_mask).mean()

            avg_class_prob = student.mean(dim=0)
            equalize_class_loss = F.binary_cross_entropy(
                avg_class_prob, torch.tensor([1.0 / 10] * 10).to(self.device)
            )
            equalize_class_loss = equalize_class_loss * unsup_mask.mean(dim=0)
            equalize_class_loss = equalize_class_loss.mean() * 10

            unsup_loss += equalize_class_loss * 0.5

            return unsup_loss

        def train_step(
            source_images,
            source_labels,
            target_images,
            target_labels,
        ):
            # target classes are only used for intermediate evaluation

            source_images = source_images.to(self.device)
            source_labels = source_labels.to(self.device)
            target_images = target_images.to(self.device)
            target_labels = target_labels.to(self.device)

            self.student_optimizer.zero_grad()
            self.student.train()
            self.teacher.train()

            source_logits = self.student(source_images)[1]
            student_target_logits = self.student(target_images)[1]
            student_target_pred = F.softmax(student_target_logits, dim=1)
            teacher_target_logits = self.teacher(target_images)[1]
            teacher_target_pred = F.softmax(teacher_target_logits, dim=1)

            clf_loss = F.cross_entropy(source_logits, source_labels)
            unsup_loss = unsup_loss_fn(student_target_pred, teacher_target_pred)

            loss_expr = clf_loss + unsup_loss * 3

            loss_expr.backward()
            self.student_optimizer.step()
            self.teacher_optimizer.step()

            source_classes = torch.argmax(F.softmax(source_logits, dim=1), dim=-1)
            target_classes = torch.argmax(student_target_pred, dim=-1)

            source_acc = accuracy_score(source_classes.cpu(), source_labels.cpu())
            target_acc = accuracy_score(target_classes.cpu(), target_labels.cpu())

            return source_acc, target_acc

        dataset_len = min(len(source_dataset), len(target_dataset))

        @fnlog("Epoch {0} of " + str(EPOCHS))
        def train_epoch(epoch_no):
            compound_dataset = enumerate(
                zip(
                    source_dataset,
                    target_dataset,
                )
            )

            source_accs = []
            target_accs = []

            for (
                batch_idx,
                (source_batch, target_batch),
            ) in compound_dataset:
                source_acc, target_acc = train_step(
                    source_batch[0],
                    source_batch[1],
                    target_batch[0],
                    target_batch[1],
                )
                source_accs.append(source_acc)
                target_accs.append(target_acc)

                source_total = sum(source_accs) / len(source_accs)
                target_total = sum(target_accs) / len(target_accs)
                percent = int(batch_idx * 100.0 / dataset_len)

                dlog(
                    f"{percent}%"
                    f" - source acc: {source_total:.3f}"
                    f" - target acc: {target_total:.3f}",
                    same_line=True,
                )

                if CLIP_DATASETS:
                    break

            return source_accs, target_accs

        def get_accuracy_on(dataset):
            accuracies = []
            for images, labels in dataset:
                images = images.to(self.device)
                labels = labels.to(self.device)
                classes = torch.argmax(F.softmax(self.student(images)[1], dim=-1), dim=1)
                accuracy = accuracy_score(classes.cpu(), labels.cpu())
                accuracies.append(accuracy)
            return sum(accuracies) / len(accuracies)

        s_accs, t_accs, sv_accs, tv_accs = [], [], [], []

        for epoch_no in range(EPOCHS):
            s_acc, t_acc = train_epoch(epoch_no + 1)
            s_accs.append(s_acc)
            t_accs.append(t_acc)
            sv_accs.append(get_accuracy_on(source_val_dataset))
            tv_accs.append(get_accuracy_on(target_val_dataset))

        return s_accs, t_accs, sv_accs, tv_accs

    def latents(self, images):
        return self.student(images)[0]


@fnlog("Graphing the results")
def graph_accuracies(svhn_train, svhn_test, mnist_train, mnist_test):
    total_svhn_train = []
    total_mnist_train = []

    epoch_line_ctr = 0
    epoch_lines = []

    for accuracies in svhn_train:
        total_svhn_train.extend(accuracies)
        epoch_line_ctr += len(accuracies)
        epoch_lines.append(epoch_line_ctr)
    for accuracies in mnist_train:
        total_mnist_train.extend(accuracies)

    x = list(range(1, len(total_svhn_train) + 1))
    plt.figure()
    plt.plot(x, total_svhn_train, label="SVHN train", color="red")
    plt.legend()
    plt.figure()
    plt.plot(x, total_mnist_train, label="MNIST train", color="blue")
    plt.legend()

    plt.figure()
    plt.plot(epoch_lines, svhn_test, label="SVHN test", color="red", ls="--")
    plt.legend()
    plt.figure()
    plt.plot(epoch_lines, mnist_test, label="MNIST test", color="blue", ls="--")
    plt.legend()

    plt.show()


@fnlog("Building a domain representation")
def tsne(model: EnsembleDA, dataset_a, dataset_b, device):
    latents = []
    classes = []
    styles = []

    for images, labels in dataset_a:
        images = images.to(device)
        latents.extend(model.latents(images))
        classes.extend(labels)
        styles.extend([0 for _ in range(len(labels))])
        if len(latents) > 200:
            break

    for images, labels in dataset_b:
        images = images.to(device)
        latents.extend(model.latents(images))
        classes.extend(labels)
        styles.extend([1 for _ in range(len(labels))])
        if len(latents) > 200:
            break

    latents = np.array([np.array(l.detach().cpu()) for l in latents])
    latents = latents.reshape((-1, 128 * 4 * 4))
    tsne = TSNE(perplexity=50, n_iter=2000, learning_rate=10, verbose=1)
    tsne_results = tsne.fit_transform(latents)

    xs = tsne_results[:, 0]
    ys = tsne_results[:, 1]

    plt.figure()
    sns.scatterplot(
        x=xs, y=ys, legend="full", alpha=0.7, palette=sns.color_palette("hls", 10),
        hue=[int(x.cpu()) for x in classes], style=styles
    )


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def main():
    torch.manual_seed(31337)

    mnist_train, mnist_test, svhn_train, svhn_test = load_and_transform_datasets()

    if TRAIN_MODEL:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu:0")
        model = EnsembleDA(device)
        s_accs, t_accs, sv_accs, tv_accs = model.train(
            svhn_train, mnist_train, svhn_test, mnist_test
        )
        torch.save((s_accs, t_accs, sv_accs, tv_accs), "accs.pth")
    else:
        s_accs, t_accs, sv_accs, tv_accs = torch.load("accs.pth")
    graph_accuracies(s_accs, sv_accs, t_accs, tv_accs)
    tsne(model, svhn_test, mnist_test, device)
    multipage("graphs.pdf")
    plt.show()

    #  tt_acc = model.evaluate(mnist_test)


if __name__ == "__main__":
    try:
        main()
    except:
        import traceback

        traceback.print_exc()
