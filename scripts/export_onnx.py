import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO


class PPOActor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observations):
        features = self.policy.extract_features(observations)
        if isinstance(features, tuple):
            latent_pi = features[0]
            latent_pi, _ = self.policy.mlp_extractor(latent_pi)
        else:
            latent_pi, _ = self.policy.mlp_extractor(features)
        logits = self.policy.action_net(latent_pi)
        return torch.argmax(logits, dim=1)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main():
    parser = argparse.ArgumentParser(description="Export a Stable-Baselines3 PPO policy to ONNX")
    parser.add_argument("--model", default="models/ppo_park_final", help="Path to .zip PPO model")
    parser.add_argument("--output", default="models/ppo_park_final.onnx", help="Output ONNX path")
    args = parser.parse_args()

    model_path = resolve(args.model)
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(model_path), device="cpu")
    model.policy.eval()
    actor = PPOActor(model.policy).eval()

    dummy_observation = torch.zeros((1, int(model.observation_space.shape[0])), dtype=torch.float32)
    torch.onnx.export(
        actor,
        dummy_observation,
        str(output_path),
        dynamo=False,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    test_observation = np.zeros((2, int(model.observation_space.shape[0])), dtype=np.float32)
    onnx_actions = session.run(["action"], {"observation": test_observation})[0]
    sb3_actions = np.array(
        [int(model.predict(test_observation[i], deterministic=True)[0]) for i in range(2)]
    )
    if not np.array_equal(onnx_actions.reshape(-1), sb3_actions):
        raise RuntimeError("ONNX action mismatch after export")

    print(f"[onnx] exported -> {output_path}")
    print("[onnx] input: observation, shape [batch, 13], float32")
    print("[onnx] output: action, shape [batch], int64; values 0-5")


if __name__ == "__main__":
    main()
