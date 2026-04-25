from __future__ import annotations

from portfolio.data.database_manager import DBManager
from portfolio.pipeline.pipeline import BaseStep
from portfolio.utils.pd_pl_utils import ensure_polars

class DBLoadAll(BaseStep):
    def __init__(self, db_manager: DBManager):
        super().__init__()
        self.db_manager = db_manager

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        self._require_state(state, require_attrs=("df",), require_non_none=("df",))

        state.df = self.db_manager.load_all()
        state.df = ensure_polars(state.df)
        self.log_out = f"df.shape: {state.df.shape}"

        return state
    

    

