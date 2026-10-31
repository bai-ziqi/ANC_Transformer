"""Compare four fixed-filter methods on EXACTLY the same timed episodes."""
from experiments.common import evaluate, load_model, save_evaluation
from utils import common_parser, load_config


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model, bundle, checkpoint = load_model(cfg)
    methods = ("transformer", "scene_wiener", "pooled_wiener", "mean_wiener")
    rows = evaluate(cfg, model, bundle, args.split, methods)
    save_evaluation(cfg, rows, args.split, "comparison", checkpoint)


if __name__ == "__main__":
    main()
