import hashlib
import torch


def digest(modules):
    h = hashlib.sha256()
    for i, m in enumerate(modules):
        for n, p in m.state_dict().items():
            h.update(f"{i}:{n}".encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Backend:
    def __init__(self, name, task, weights, estimators=2, seed=42, device="cuda:0"):
        self.name = name
        self.task = task
        self.estimators = estimators
        self.seed = seed
        self.device = device
        self.modules = []
        self.initial_hash = None
        if name == "tabpfn35":
            from tabpfn import TabPFNClassifier, TabPFNRegressor

            cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
            self.model = cls(
                model_path=weights,
                device=device,
                n_estimators=estimators,
                auto_scale_n_estimators=False,
                random_state=seed,
                inference_precision=torch.float32,
                differentiable_input=True,
                ignore_pretraining_limits=True,
                n_preprocessing_jobs=1,
                enable_torch_compile=False,
            )
        elif name == "tabfm":
            from tabfm.src.pytorch.tabfm_v1_0_0 import load

            self.model = load(
                model_type=task,
                checkpoint_path=str(weights),
                device=device,
                dtype=None,
                use_cache=False,
            )
            self.modules = [self.model]
            self.freeze()
            from torch.utils.checkpoint import checkpoint

            self.checkpointed_blocks = 0

            def wrap(original):
                def forward(*args, **kwargs):
                    if torch.is_grad_enabled() and any(
                        torch.is_tensor(x) and x.requires_grad for x in args
                    ):
                        return checkpoint(
                            original,
                            *args,
                            use_reentrant=False,
                            preserve_rng_state=True,
                            **kwargs,
                        )
                    return original(*args, **kwargs)

                return forward

            for module in self.model.modules():
                if module.__class__.__name__ == "MultiheadAttentionBlock":
                    module.forward = wrap(module.forward)
                    self.checkpointed_blocks += 1
            assert self.checkpointed_blocks > 0
            print("CHECKPOINTED_BLOCKS", self.checkpointed_blocks, flush=True)
        else:
            raise ValueError(name)

    def freeze(self):
        for m in self.modules:
            m.requires_grad_(False)
            m.eval()
        self.initial_hash = digest(self.modules)

    def predict(self, xc, yc, xq):
        if self.name == "tabpfn35":
            self.model.fit_with_differentiable_input(xc, yc)
            if not self.modules:
                self.modules = list(self.model.models_)
                self.freeze()
            elif any(
                old is not new
                for old, new in zip(self.modules, self.model.models_, strict=True)
            ):
                raise RuntimeError(
                    "TabPFN replaced its frozen model instances during fitting."
                )
            assert all(
                not p.requires_grad for m in self.modules for p in m.parameters()
            )
            if self.task == "classification":
                return self.model.forward(
                    xq, use_inference_mode=True, return_logits=True
                )
            _, outputs, _ = self.model.forward(xq, use_inference_mode=True)
            pred = [
                self.model.znorm_space_bardist_.mean(v) * self.model.y_train_std_
                + self.model.y_train_mean_
                for v in outputs
            ]
            return torch.stack(pred).mean(0)
        n = len(xc)
        x = torch.cat([xc, xq])
        mu = yc.float().mean()
        std = yc.float().std(correction=0).clamp_min(1e-6)
        context = yc if self.task == "classification" else (yc - mu) / std
        y = torch.cat(
            [context, torch.zeros(len(xq), device=yc.device, dtype=context.dtype)]
        )
        sizes = torch.tensor([n], device=x.device)
        # Raw scalar pseudo-columns enter the original cell embedder alongside table columns.
        outputs = []
        for member in range(self.estimators):
            generator = torch.Generator(device="cpu").manual_seed(self.seed + member)
            order = torch.randperm(x.shape[1], generator=generator).to(x.device)
            outputs.append(
                self.model(x[:, order].unsqueeze(0), y.unsqueeze(0), sizes)[0, n:]
            )
        out = torch.stack(outputs).mean(0)
        if self.task == "classification":
            return out[:, : int(yc.max().item()) + 1]
        return out.squeeze(-1) * std + mu

    def assert_frozen(self):
        assert self.modules and digest(self.modules) == self.initial_hash
        assert all(
            not p.requires_grad and p.grad is None
            for m in self.modules
            for p in m.parameters()
        )
