"""One Word report for a whole project: data, preparation, methods, results, an activity log and references.

The results section reuses the report content each module built when its result was saved (the same tables and
charts as that module's own Word export). Results saved before that content was kept get a short summary of their
settings instead, with a note saying how to export the full tables.
"""

from datetime import datetime

from pydantic import ValidationError

from . import methods_text
from .reports import ReportIn, ReportSection, ReportTable

MODULE_NAMES = {
    "descriptive": "Descriptive statistics", "correlation": "Correlation", "parametric": "Parametric test",
    "nonparametric": "Non-parametric test", "anova": "ANOVA", "regression": "Regression", "pca": "PCA",
    "clustering": "Clustering", "trend": "Trend analysis", "gxe": "Genotype × environment", "divergence": "Genetic divergence",
    "timeseries": "Time series", "sem": "Structural equation model", "imputation": "Multiple imputation",
}


def _when(value):
    return value.strftime("%d %b %Y, %H:%M") if isinstance(value, datetime) else "–"


def _parameter_rows(parameters):
    rows = []
    for key, value in (parameters or {}).items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            shown = ", ".join(str(v.get("name", v)) if isinstance(v, dict) else str(v) for v in value)
        elif isinstance(value, dict):
            shown = ", ".join(f"{k}: {v}" for k, v in value.items() if v not in (None, ""))
        else:
            shown = str(value)
        rows.append([key.replace("_", " ").capitalize(), shown[:500]])
    return rows


def _stored_sections(report):
    """The sections a module built for its own Word export, as sub-sections; None when unusable."""
    if not isinstance(report, dict) or not isinstance(report.get("sections"), list):
        return None
    sections = []
    for raw in report["sections"]:
        try:
            section = ReportSection.model_validate({**raw, "level": 3})
        except (ValidationError, TypeError):
            continue
        sections.append(section)
    notes = [str(n) for n in report.get("notes") or [] if n]
    if notes:
        sections.append(ReportSection(heading="Notes", level=3, bullets=notes[:30]))
    return sections or None


def build_project_report(project, datasets, results, recipes, *, title=None, include_activity=True,
                         include_references=True):
    """The ReportIn for a project; `results` must have their parameters, result and report loaded."""
    references = []
    sections = []

    # 1. Data
    if datasets:
        sections.append(ReportSection(
            heading="Data",
            paragraphs=[f"The project holds {len(datasets)} dataset{'s' if len(datasets) != 1 else ''}."],
            tables=[ReportTable(columns=["File", "Rows", "Columns", "Uploaded"],
                                rows=[[d.filename, d.n_rows, d.n_columns, _when(d.created_at)] for d in datasets])],
        ))

    # 2. Data preparation
    if recipes:
        prep = ReportSection(heading="Data preparation")
        for recipe in recipes:
            sentences, keys = methods_text.describe_recipe(recipe.steps)
            references += keys
            source = f" (written for {recipe.source_dataset})" if recipe.source_dataset else ""
            prep.paragraphs.append(f"Recipe \"{recipe.name}\"{source}: " + " ".join(
                f"({i}) {sentence}" for i, sentence in enumerate(sentences, start=1)))
            if recipe.description:
                prep.paragraphs.append(recipe.description)
        sections.append(prep)

    # 3. Statistical methods
    if results:
        methods = ReportSection(heading="Statistical methods")
        for result in results:
            described = methods_text.describe_analysis(result.module, result.parameters, result.result)
            references += described["references"]
            methods.paragraphs.append(f"{result.title}. {described['paragraph']}")
        software, keys = methods_text.software_paragraph()
        references += keys
        methods.paragraphs.append(software)
        sections.append(methods)

    # 4. Results, one block per analysis
    for index, result in enumerate(results, start=1):
        module_name = MODULE_NAMES.get(result.module, result.module)
        sections.append(ReportSection(
            heading=f"Result {index}: {result.title}",
            paragraphs=[f"{module_name} · saved {_when(result.created_at)}"],
        ))
        stored = _stored_sections(result.report)
        if stored:
            sections.extend(stored)
        else:
            rows = _parameter_rows(result.parameters)
            sections.append(ReportSection(
                heading="Settings",
                level=3,
                tables=[ReportTable(columns=["Setting", "Value"], rows=rows)] if rows else [],
                paragraphs=["The detailed tables were not stored with this result (it was saved before project reports "
                            "kept them). Open it from the Projects page and save it again, or export the Word report "
                            "from its module."],
            ))

    # 5. Activity log
    if include_activity:
        events = [(d.created_at, "Dataset uploaded", d.filename) for d in datasets]
        events += [(r.created_at, "Analysis saved", f"{r.title} ({MODULE_NAMES.get(r.module, r.module)})") for r in results]
        for recipe in recipes:
            events.append((recipe.created_at, "Recipe saved", recipe.name))
            if recipe.updated_at and recipe.created_at and (recipe.updated_at - recipe.created_at).total_seconds() > 1:
                events.append((recipe.updated_at, "Recipe updated", recipe.name))
        events.sort(key=lambda event: event[0] or datetime.min)
        if events:
            sections.append(ReportSection(
                heading="Activity log",
                paragraphs=["What was stored in this project, in order (times are server time, UTC)."],
                tables=[ReportTable(columns=["When", "Event", "Item"], rows=[[_when(w), e, i] for w, e, i in events])],
            ))

    # 6. References
    if include_references and references:
        sections.append(ReportSection(
            heading="References",
            paragraphs=methods_text.reference_list(references),
        ))

    if not sections:
        sections.append(ReportSection(heading="Summary", paragraphs=["This project has no datasets, recipes or saved analyses yet."]))

    return ReportIn(
        title=(title or f"{project.name} – analysis report")[:200],
        test_applied=f"{len(results)} analys{'is' if len(results) == 1 else 'es'}, {len(recipes)} cleaning recipe{'s' if len(recipes) != 1 else ''}",
        dataset_name=", ".join(d.filename for d in datasets)[:255] or None,
        filename=f"StatSathi_{project.name}_report"[:150],
        sections=sections,
        notes=["Methods paragraphs are generated from the saved settings; read them against what you did and edit as "
               "needed. References are in APA style without DOIs - check them against your journal's style."],
    )
