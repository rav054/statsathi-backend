import io
import math
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/sem", tags=["SEM"])

def sanitize_float(val, default=None):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (ValueError, TypeError):
        return default

@router.get("/templates")
def get_templates():
    return {
        "simple_regression": "Y ~ X1 + X2 + X3",
        "mediation": "Y ~ X + M\nM ~ X",
        "latent_factor": "Latent =~ Item1 + Item2 + Item3\nY ~ Latent + X1",
        "full_sem": "Latent1 =~ A1 + A2 + A3\nLatent2 =~ B1 + B2\nY ~ Latent1 + Latent2 + Control"
    }

@router.post("/fit")
async def fit_sem(file: UploadFile = File(...), model: str = Form(...)):
    try:
        import semopy
    except ImportError:
        raise HTTPException(status_code=500, detail="semopy library is not installed on backend.")

    # 1. Parse CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    # Clean missing values
    df_clean = df.dropna(how="all").copy()

    # 2. Fit Model
    try:
        mod = semopy.Model(model)
        mod.fit(df_clean)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SEM model fitting failed: {str(e)}")

    # 3. Fit Indices Calculation
    fit_indices = {
        "CFI": None,
        "RMSEA": None,
        "SRMR": None,
        "AIC": None,
        "BIC": None,
        "chi2": None,
        "df": None,
        "pvalue": None
    }
    
    try:
        stats = semopy.calc_stats(mod)
        if isinstance(stats, pd.DataFrame):
            stats_dict = {}
            for col in stats.columns:
                val = stats[col].iloc[0] if len(stats[col]) > 0 else None
                stats_dict[str(col).upper()] = val
                stats_dict[str(col)] = val
        elif isinstance(stats, dict):
            stats_dict = {str(k).upper(): v for k, v in stats.items()}
            stats_dict.update(stats)
        else:
            stats_dict = {}

        def extract_stat(keys):
            for k in keys:
                if k in stats_dict and stats_dict[k] is not None:
                    return sanitize_float(stats_dict[k])
            return None

        fit_indices["CFI"] = extract_stat(["CFI", "cfi"])
        fit_indices["RMSEA"] = extract_stat(["RMSEA", "rmsea"])
        fit_indices["SRMR"] = extract_stat(["SRMR", "srmr"])
        fit_indices["AIC"] = extract_stat(["AIC", "aic"])
        fit_indices["BIC"] = extract_stat(["BIC", "bic"])
        fit_indices["chi2"] = extract_stat(["CHI2", "CHI2 P-VALUE", "chi2", "Chi2", "CHI-SQUARE"])
        fit_indices["df"] = extract_stat(["DOF", "DF", "DoF", "df"])
        fit_indices["DoF"] = fit_indices["df"]
        fit_indices["pvalue"] = extract_stat(["CHI2 P-VALUE", "P-VALUE", "PVALUE", "pvalue", "p-value"])
    except Exception:
        pass

    # 4. Parameters Inspection
    parameters = []
    try:
        inspect_df = mod.inspect()
        for _, row in inspect_df.iterrows():
            lval = str(row.get("lval", row.get("lvalue", "")))
            rval = str(row.get("rval", row.get("rvalue", "")))
            op = str(row.get("op", "~"))

            est = sanitize_float(row.get("Estimate", row.get("Est", row.get("est", None))))
            se = sanitize_float(row.get("Std. Err", row.get("Std.Err", row.get("se", None))))
            z = sanitize_float(row.get("z-value", row.get("z_value", row.get("z", None))))
            p = sanitize_float(row.get("p-value", row.get("p_value", row.get("pvalue", None))))

            sig = bool(p is not None and p < 0.05)

            parameters.append({
                "lval": lval,
                "rval": rval,
                "op": op,
                "Estimate": est if est is not None else 0.0,
                "Std.Err": se if se is not None else 0.0,
                "z-value": z if z is not None else 0.0,
                "p-value": p if p is not None else 1.0,
                "significant": sig
            })
    except Exception:
        parameters = []

    n_obs = len(df_clean)

    return {
        "success": True,
        "fit_indices": fit_indices,
        "parameters": parameters,
        "model_syntax": model,
        "n_obs": n_obs
    }

@router.post("/diagram")
async def get_sem_diagram(file: UploadFile = File(...), model: str = Form(...)):
    try:
        import semopy
    except ImportError:
        raise HTTPException(status_code=500, detail="semopy library is not installed on backend.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents)).dropna(how="all")
        mod = semopy.Model(model)
        mod.fit(df)

        try:
            g = semopy.semplot(mod, "temp_sem_diagram.gv")
            if hasattr(g, 'pipe'):
                svg_data = g.pipe(format='svg').decode('utf-8')
                return JSONResponse(content={"success": True, "svg": svg_data})
        except Exception as e:
            return JSONResponse(content={"success": False, "error": str(e), "svg": None})

        return JSONResponse(content={"success": False, "error": "Diagram rendering not supported", "svg": None})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e), "svg": None})
