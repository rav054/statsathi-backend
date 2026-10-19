"""Analysis API: mounts the per-module routers from app/modules under /analyze."""
from fastapi import APIRouter

from .modules import (data_io, correlation, parametric, nonparametric, anova, plots, pca, descriptive, clustering, sem,
                      regression, transform, result_charts, trend, gxe, divergence, timeseries, figure_composer, dataprep)

router = APIRouter(prefix="/analyze", tags=["Analysis"])

for module in (data_io, correlation, parametric, nonparametric, anova, plots, pca, descriptive, clustering, sem,
               regression, transform, result_charts, trend, gxe, divergence, timeseries, figure_composer, dataprep):
    router.include_router(module.router)
