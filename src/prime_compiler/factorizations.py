"""Factorization database — maps torch ops to their computational primes."""

from prime_compiler.primes import Prime as P, MissingPrime as X, Status

# Each entry: (primes, missing_primes, status, note)
# primes: set of Prime enums used
# missing: set of MissingPrime enums needed (empty = fully mappable)

FACTORIZATIONS = {
    # ── Transformer / LLM Core ──
    "linear":              ({P.P1, P.P2},           set(),     Status.M, "MatMul via crossbar (P1·P2)"),
    "matmul":              ({P.P1, P.P2},           set(),     Status.M, "MatMul via crossbar"),
    "mm":                  ({P.P1, P.P2},           set(),     Status.M, "MatMul via crossbar"),
    "bmm":                 ({P.P1, P.P2},           set(),     Status.M, "Batched MatMul via crossbar"),
    "addmm":               ({P.P1, P.P2},           set(),     Status.M, "Bias + MatMul"),
    "scaled_dot_product_attention": ({P.P11, P.P2, P.P3, P.P1, P.P12}, set(), Status.M, "QK^T/√d softmax attn·V"),
    "softmax":             ({P.P3, P.P1, P.P12},    set(),     Status.M, "exp + sum + div; or Race (P3·P5)"),
    "log_softmax":         ({P.P3, P.P1, P.P12, P.P4}, set(),  Status.M, "softmax + log"),
    "layer_norm":          ({P.P1, P.P2, P.P11, P.P12}, set(), Status.M, "AGC feedback or translinear pipeline"),
    "rms_norm":            ({P.P1, P.P2, P.P11, P.P12}, set(), Status.M, "Solved: AGC feedback (~12ns)"),
    "group_norm":          ({P.P1, P.P2, P.P11, P.P12}, set(), Status.M, "Stats over fixed channel groups"),
    "instance_norm":       ({P.P1, P.P2, P.P11, P.P12}, set(), Status.M, "GroupNorm with group=1"),
    "batch_norm":          ({P.P1, P.P2, P.P11, P.P12, P.P8}, {X.X2}, Status.G, "Training: cross-sample stats need X2; inference: M"),
    "embedding":           ({P.P8, P.P2},           set(),     Status.M, "Crossbar with one-hot"),

    # ── Activation Functions ──
    "relu":                ({P.P7, P.P11},          set(),     Status.M, "Comparator + analog switch"),
    "leaky_relu":          ({P.P7, P.P11, P.P2, P.P1}, set(), Status.M, "Two scaled branches gated by P7"),
    "prelu":               ({P.P7, P.P11, P.P2, P.P1}, set(), Status.M, "Inference: learned α fixed"),
    "elu":                 ({P.P7, P.P11, P.P3, P.P1, P.P2}, set(), Status.M, "Neg branch: α(eˣ-1)"),
    "selu":                ({P.P7, P.P11, P.P3, P.P1, P.P2}, set(), Status.M, "Scaled ELU"),
    "gelu":                ({P.P11, P.P6, P.P2},    set(),     Status.M, "Approx x·σ(1.702x): Gilbert×Sigmoid"),
    "silu":                ({P.P11, P.P6},           set(),     Status.M, "x·σ(x): Gilbert × FET-pair"),
    "sigmoid":             ({P.P6},                  set(),     Status.M, "Single subthreshold FET pair"),
    "tanh":                ({P.P6, P.P2, P.P1},     set(),     Status.M, "2σ(2x)-1: differential FET pair"),
    "hardswish":           ({P.P11, P.P7, P.P1, P.P2}, set(), Status.M, "x·HardSigmoid(x)"),
    "hardsigmoid":         ({P.P7, P.P1, P.P2},     set(),    Status.M, "clip(x/6+0.5,0,1)"),
    "hardtanh":            ({P.P7},                  set(),     Status.M, "Double threshold clamp"),
    "softplus":            ({P.P4, P.P1, P.P3},     set(),     Status.M, "ln(1+eˣ)"),
    "mish":                ({P.P11, P.P6, P.P3, P.P1, P.P4}, set(), Status.M, "x·tanh(softplus(x))"),
    "log_sigmoid":         ({P.P6, P.P4},           set(),     Status.M, "ln(σ(x))"),
    "threshold":           ({P.P7},                  set(),     Status.M, "Comparator"),

    # ── Pooling ──
    "avg_pool1d":          ({P.P1, P.P2},           set(),     Status.M, "Resistor averaging"),
    "avg_pool2d":          ({P.P1, P.P2},           set(),     Status.M, "Resistor averaging"),
    "avg_pool3d":          ({P.P1, P.P2},           set(),     Status.M, "Resistor averaging"),
    "adaptive_avg_pool1d": ({P.P1, P.P2},           set(),     Status.M, "Fixed output size = fixed topology"),
    "adaptive_avg_pool2d": ({P.P1, P.P2},           set(),     Status.M, "Fixed output size = fixed topology"),
    "max_pool1d":          ({P.P5},                  set(),     Status.M, "WTA over receptive field"),
    "max_pool2d":          ({P.P5},                  set(),     Status.M, "WTA over receptive field"),
    "max_pool3d":          ({P.P5},                  set(),     Status.M, "WTA over receptive field"),

    # ── Convolution ──
    "conv1d":              ({P.P1, P.P11, P.P8},    set(),     Status.M, "Crossbar with sliding window"),
    "conv2d":              ({P.P1, P.P11, P.P8},    set(),     Status.M, "2D crossbar tiling"),
    "conv3d":              ({P.P1, P.P11, P.P8},    set(),     Status.M, "3D crossbar tiling"),
    "conv_transpose1d":    ({P.P1, P.P11, P.P8},    set(),     Status.M, "Transposed = routing change"),
    "conv_transpose2d":    ({P.P1, P.P11, P.P8},    set(),     Status.M, "Transposed = routing change"),

    # ── Arithmetic (element-wise) ──
    "add":                 ({P.P1},                  set(),     Status.M, "KCL at node"),
    "sub":                 ({P.P1},                  set(),     Status.M, "KCL (inverted)"),
    "mul":                 ({P.P11},                 set(),     Status.M, "Gilbert cell"),
    "div":                 ({P.P12},                 set(),     Status.M, "Translinear loop"),
    "neg":                 ({P.P2},                  set(),     Status.M, "Inverter (scale by -1)"),
    "abs":                 ({P.P7, P.P11},           set(),     Status.M, "Full-wave rectifier"),
    "reciprocal":          ({P.P12},                 set(),     Status.M, "Translinear 1/x"),
    "sqrt":                ({P.P12},                 set(),     Status.M, "Translinear √x"),
    "rsqrt":               ({P.P12},                 set(),     Status.M, "Translinear 1/√x"),
    "exp":                 ({P.P3},                  set(),     Status.M, "Subthreshold FET"),
    "log":                 ({P.P4},                  set(),     Status.M, "Diode voltage"),
    "pow":                 ({P.P3, P.P4, P.P2},     set(),     Status.M, "exp(b·ln(x))"),
    "clamp":               ({P.P7},                  set(),     Status.M, "Dual comparator"),
    "maximum":             ({P.P5},                  set(),     Status.M, "WTA"),
    "minimum":             ({P.P5},                  set(),     Status.M, "Inverted WTA"),

    # ── Reduction ──
    "sum":                 ({P.P1},                  set(),     Status.M, "KCL"),
    "mean":                ({P.P1, P.P2},            set(),     Status.M, "KCL + resistor divider"),
    "amax":                ({P.P5},                  set(),     Status.M, "WTA"),
    "amin":                ({P.P5},                  set(),     Status.M, "Inverted WTA"),
    "var":                 ({P.P1, P.P2, P.P11},    set(),     Status.M, "Sum of squares / N"),
    "std":                 ({P.P1, P.P2, P.P11, P.P12}, set(), Status.M, "√var via translinear"),
    "norm":                ({P.P1, P.P11, P.P12},   set(),     Status.M, "Sum squares, sqrt"),
    "argmax":              ({P.P5},                  set(),     Status.M, "Race / WTA"),
    "argmin":              ({P.P5},                  set(),     Status.M, "Inverted race"),

    # ── Loss Functions ──
    "cross_entropy":       ({P.P3, P.P4, P.P1, P.P2}, set(),  Status.M, "Solved: race+diode+KCL"),
    "nll_loss":            ({P.P1, P.P5},            set(),     Status.M, "Select + negate"),
    "mse_loss":            ({P.P1, P.P2, P.P11},    set(),     Status.M, "Subtract, square, sum, scale"),
    "l1_loss":             ({P.P1, P.P2, P.P7, P.P11}, set(), Status.M, "Full-wave rectifier loss"),
    "kl_div":              ({P.P1, P.P4, P.P11},    set(),     Status.M, "Σp·ln(p/q) via diodes"),
    "binary_cross_entropy":({P.P4, P.P1, P.P11},    set(),     Status.M, "Log + multiply + sum"),
    "cosine_similarity":   ({P.P1, P.P11, P.P12},   set(),     Status.M, "Dot / (norm×norm)"),
    "triplet_margin_loss": ({P.P1, P.P2, P.P7, P.P11, P.P12}, set(), Status.M, "Distance + margin + clamp"),
    "ctc_loss":            ({P.P1, P.P3, P.P4, P.P11}, {X.X2, X.X3}, Status.U, "Dynamic alignment graph"),

    # ── Dropout / Stochastic ──
    "dropout":             ({P.P7, P.P10},           set(),     Status.M, "Random mask × threshold"),
    "alpha_dropout":       ({P.P7, P.P10, P.P2, P.P1}, set(), Status.M, "Scaled dropout"),

    # ── Shape ops (no computation) ──
    "reshape":             (set(),                   set(),     Status.M, "Routing only"),
    "permute":             (set(),                   set(),     Status.M, "Routing only"),
    "transpose":           (set(),                   set(),     Status.M, "Routing only"),
    "flatten":             (set(),                   set(),     Status.M, "Routing only"),
    "unsqueeze":           (set(),                   set(),     Status.M, "Routing only"),
    "squeeze":             (set(),                   set(),     Status.M, "Routing only"),
    "cat":                 ({P.P1},                  set(),     Status.M, "Wire concatenation"),
    "split":               (set(),                   set(),     Status.M, "Routing only"),
    "chunk":               (set(),                   set(),     Status.M, "Routing only"),
    "select":              (set(),                   set(),     Status.M, "Routing only"),
    "slice":               (set(),                   set(),     Status.M, "Routing only"),
    "getitem":             (set(),                   set(),     Status.M, "Routing only"),

    # ── Unmappable ──
    "sort":                ({P.P5},                  {X.X2},    Status.G, "Iterated WTA needs dynamic routing"),
    "topk":                ({P.P5, P.P3},            set(),     Status.M, "K-fold race"),
    "unique":              (set(),                   {X.X1},    Status.U, "Symbol comparison"),
    "scatter":             ({P.P1},                  {X.X2},    Status.G, "Dynamic indexing"),
    "gather":              (set(),                   {X.X2},    Status.G, "Dynamic indexing"),
    "index_select":        (set(),                   {X.X2},    Status.G, "Dynamic indexing"),
}


def get_factorization(op_name: str):
    """Look up factorization for a torch op name."""
    clean = op_name.lower().replace("torch.", "").replace("nn.functional.", "")
    clean = clean.replace("aten::", "").replace("_", "").rstrip("_")

    # Try exact match first
    for key, val in FACTORIZATIONS.items():
        if clean == key.replace("_", ""):
            return key, val

    # Try substring match
    for key, val in FACTORIZATIONS.items():
        if key.replace("_", "") in clean or clean in key.replace("_", ""):
            return key, val

    return None, None


def classify(primes: set, missing: set) -> Status:
    """Determine mappability status from prime sets."""
    if not missing:
        return Status.M
    from prime_compiler.primes import GPU_AVAILABLE
    if missing.issubset(GPU_AVAILABLE):
        return Status.G
    return Status.U
