"""Config sanity checks."""

import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)


def check_prior_preservation_config(cfg: SimpleNamespace) -> None:
    """
    Sanity: prior preservation is only active when explicitly configured via loss class.

    Prior preservation (DreamBooth-style instance/prior split) happens only when
    loss.class_name is FlowMatchingWithPriorPreservation. Logs when that is the case.
    """
    loss_cfg = getattr(cfg, "loss", None)
    if loss_cfg is None:
        return
    class_name = getattr(loss_cfg, "class_name", None)
    if class_name is None:
        return
    name = str(class_name).lower()
    if "priorpreservation" in name.replace("_", "").replace(".", ""):
        logger.info(
            "Prior preservation enabled via loss config: %s",
            class_name,
        )
