from __future__ import annotations

import numpy as np
import pandas as pd
from chemprop.data import MoleculeDatapoint, MoleculeDataset, MulticomponentDataset
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

from lnpml.representations.base import RepresentationBuilder


class ChempropGraphBuilder(RepresentationBuilder):
    """Build a chemprop MoleculeDataset or MulticomponentDataset from a DataFrame.

    For a single SMILES column the result is a plain MoleculeDataset.
    For multiple SMILES columns the result is a MulticomponentDataset where each
    component is a separate MoleculeDataset (one per lipid type).

    Extra feature columns (X_d) are attached to the first component's datapoints
    and concatenated after graph aggregation by chemprop's FFN.
    """

    def build(
        self,
        df: pd.DataFrame,
        smiles_cols: list[str],
        target_col: str | None = None,
        extra_feature_cols: list[str] | None = None,
    ) -> MoleculeDataset | MulticomponentDataset:
        # target_col may be absent for prediction-only data (no labels).
        targets: np.ndarray | None = None
        if target_col is not None and target_col in df.columns:
            targets = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

        x_d: np.ndarray | None = None
        if extra_feature_cols:
            x_d = df[extra_feature_cols].to_numpy(dtype=np.float32)

        featurizer = SimpleMoleculeMolGraphFeaturizer()

        component_datasets: list[MoleculeDataset] = []
        for i, col in enumerate(smiles_cols):
            datapoints = [
                MoleculeDatapoint.from_smi(
                    smi,
                    y=targets[j] if (i == 0 and targets is not None) else None,
                    x_d=x_d[j] if (i == 0 and x_d is not None) else None,
                )
                for j, smi in enumerate(df[col])
            ]
            # This is a list of components, where the first component has the target variable and extra features.
            # This has as many elements as components being modelled, mirroring https://github.com/chemprop/chemprop/blob/main/examples/training_regression_multicomponent.ipynb
            component_datasets.append(
                MoleculeDataset(datapoints, featurizer=featurizer)
            )

        if len(component_datasets) == 1:
            return component_datasets[0]
        return MulticomponentDataset(datasets=component_datasets)
