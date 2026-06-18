"""Feature engineering for the tree-based and neural legs.

* :mod:`store_sales.features.calendar` — holiday tables, oil, transactions,
  calendar features, and the shared national-holiday helper.
* :mod:`store_sales.features.lgbm_features` — per-(store, family) lag/rolling
  features and the leakage-aware per-horizon feature selector.
* :mod:`store_sales.features.darts_features` — preprocessing for the darts
  per-family GBT family.
"""
from __future__ import annotations
