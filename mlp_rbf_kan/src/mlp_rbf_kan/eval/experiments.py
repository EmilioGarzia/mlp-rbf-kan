import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from mlp_rbf_kan.models import *
from mlp_rbf_kan.eval.tools import *


class ExperimentCfg:
    def __init__(
        self,
        models = None,
        seeds = [32,23,54,2,45],
        input_dim = 2,
        hidden_dim = 64,
        output_dim = 1,
        centers = 8,
        domain = (0, 1),
        shared_basis = True,
        epochs = 50,
        mlp_lr = 1e-3,
        kan_lr = 0.5,
        batch_size = -1,
        mlp_optimizer = "adam",
        kan_optimizer = "lbfgs",
        device = "cuda",
        dataset = None,
        train_loader = None,
        test_loader = None
    ):
        self.models = models
        self.seeds = seeds
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.centers = centers
        self.domain = domain
        self.shared_basis = shared_basis
        self.epochs = epochs
        self.mlp_lr = mlp_lr
        self.kan_lr = kan_lr
        self.batch_size = batch_size
        self.mlp_optimizer = mlp_optimizer
        self.kan_optimizer = kan_optimizer
        self.device = device
        self.dataset = dataset
        
        if dataset is not None:
            self.x_train = dataset["train_input"].to(device)
            self.y_train = dataset["train_label"].to(device)
            self.x_test = dataset["test_input"].to(device)
            self.y_test = dataset["test_label"].to(device)
        else:
            self.x_train = None
            self.y_train = None
            self.x_test = None
            self.y_test = None
        
        self.train_loader = train_loader
        self.test_loader = test_loader


class ExperimentalRunner:
    def __init__(self, cfg, metrics=None):
        """
        metrics: metrics to track
        """
        self.cfg = cfg
        self.all_results = None

        # Default metrics
        if metrics is None:
            if self.cfg.train_loader is None:
                self.metrics = ["relative_L2", "RMSE", "L_inf"]
            else:
                self.metrics = ["accuracy", "precision", "recall", "f1"]
        else:
            self.metrics = metrics

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run_single_experiment(self, seed):
        self._set_seed(seed)

        models = {}
        for name, factory in self.cfg.models.items():
            model = factory()
            model = model.to(self.cfg.device)
            models[name] = model

        confusion_matrices = {}
        results = {}

        # --- classification ---
        if hasattr(self.cfg, "train_loader") and self.cfg.train_loader is not None:
            for name, model in models.items():
                # training
                train_classification(
                    model,
                    self.cfg.train_loader,
                    self.cfg.test_loader,
                    epochs=self.cfg.epochs,
                    lr=self.cfg.mlp_lr if name.lower() == "mlp" else self.cfg.kan_lr,
                    optimizer_name=self.cfg.mlp_optimizer if name.lower() == "mlp" else self.cfg.kan_optimizer,
                    device=self.cfg.device
                )

                # evaluation
                metrics = evaluate_classification(
                    model,
                    self.cfg.test_loader,
                    device=self.cfg.device
                )

                # Salva solo scalari come float
                results[name] = {
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"])
                }

                confusion_matrices[name] = metrics["confusion_matrix"]

        # --- regression ---
        else:
            for name, model in models.items():
                lr = self.cfg.mlp_lr if name.lower() == "mlp" else self.cfg.kan_lr
                optimizer = self.cfg.mlp_optimizer if name.lower() == "mlp" else self.cfg.kan_optimizer
                train_regression(
                    model,
                    self.cfg.x_train,
                    self.cfg.y_train,
                    self.cfg.x_test,
                    self.cfg.y_test,
                    epochs=self.cfg.epochs,
                    lr=lr,
                    batch_size=self.cfg.batch_size,
                    optimizer_type=optimizer,
                    device=self.cfg.device,
                    model_name=name
                )
                # evaluation
                metrics_dict = compute_all_metrics(
                    model(self.cfg.x_test).cpu(),
                    self.cfg.y_test.cpu()
                )
                results[name] = metrics_dict

        return results

    def run_battery_experiments(self):
        seeds = self.cfg.seeds
        all_results = None

        for seed in seeds:
            print(f"\n=== Running seed {seed} ===")
            metrics = self.run_single_experiment(seed)

            if all_results is None:
                all_results = {k: [] for k in metrics.keys()}

            for model_name, model_metrics in metrics.items():
                all_results[model_name].append(model_metrics)

        print("\n================ FINAL RESULTS (mean ± std) ================\n")

        for model_name, runs in all_results.items():
            print(model_name)

            for metric in runs[0].keys():
                values = []

                for r in runs:
                    v = r[metric]
                    if isinstance(v, (int, float, np.number)):
                        values.append(v)

                if len(values) == 0:
                    continue  # skip non-numeric metrics

                values = np.array(values, dtype=float)
                print(f"  {metric}: {values.mean():.6e} ± {values.std():.2e}")

            print()

        self.all_results = all_results
        return all_results


    def summarize_results(self, results=None):
        """
        Trasforma:
            model -> list of metric dicts
        in:
            model -> metric -> (mean, std)
        """
        if results is None:
            results = self.all_results

        summary = {}
        for model_name, runs in results.items():
            summary[model_name] = {}
            metrics = runs[0].keys()
            for metric in metrics:
                values = np.array([r[metric] for r in runs])
                summary[model_name][metric] = (values.mean(), values.std())
        return summary

    def plot_metrics_summary(self, save_path=None, figsize=(12, 4), log_scale=False):
        results = self.summarize_results()
        model_names = list(results.keys())
        x = np.arange(len(model_names))

        fig, axes = plt.subplots(1, len(self.metrics), figsize=figsize, sharex=True)
        if len(self.metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, self.metrics):
            means = [results[m][metric][0] for m in model_names]
            stds = [results[m][metric][1] for m in model_names]

            ax.errorbar(x, means, yerr=stds, fmt='o', capsize=4, linewidth=1.5)
            ax.set_title(metric.replace("_", " "))
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=20)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            if log_scale:
                ax.set_yscale("log")

        fig.suptitle("Model comparison (mean ± std over seeds)", fontsize=14)
        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        plt.show()