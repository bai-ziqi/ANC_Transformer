"""Evaluate the selected Transformer on untouched future test episodes."""
from experiments.common import evaluate, load_model, save_evaluation
from utils import common_parser, load_config


def main():
    args = common_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    model, bundle, checkpoint = load_model(cfg)
    rows = evaluate(cfg, model, bundle, "test")
    save_evaluation(cfg, rows, "test", "transformer", checkpoint)


if __name__ == "__main__":
    main()
