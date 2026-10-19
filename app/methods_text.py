"""Methods paragraphs and references for a report, written from what an analysis actually did.

``describe_analysis(module, parameters, result)`` turns a saved analysis into a paragraph for a methods section and
the keys of the references it relies on; ``describe_step(step)`` does the same for one data-preparation step. The
references are the original or standard sources for each method. They are written in APA style without DOIs:
check them against your journal's style before submission.
"""

REFERENCES = {
    # design and comparisons of means
    "gomez1984": "Gomez, K. A., & Gomez, A. A. (1984). Statistical procedures for agricultural research (2nd ed.). Wiley.",
    "tukey1949": "Tukey, J. W. (1949). Comparing individual means in the analysis of variance. Biometrics, 5(2), 99–114.",
    "kramer1956": "Kramer, C. Y. (1956). Extension of multiple range tests to group means with unequal numbers of replications. Biometrics, 12(3), 307–310.",
    "duncan1955": "Duncan, D. B. (1955). Multiple range and multiple F tests. Biometrics, 11(1), 1–42.",
    "games1976": "Games, P. A., & Howell, J. F. (1976). Pairwise multiple comparison procedures with unequal n's and/or variances: A Monte Carlo study. Journal of Educational Statistics, 1(2), 113–125.",
    "piepho2004": "Piepho, H.-P. (2004). An algorithm for a letter-based representation of all-pairwise comparisons. Journal of Computational and Graphical Statistics, 13(2), 456–466.",
    "shapiro1965": "Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). Biometrika, 52(3–4), 591–611.",
    "levene1960": "Levene, H. (1960). Robust tests for equality of variances. In I. Olkin (Ed.), Contributions to probability and statistics (pp. 278–292). Stanford University Press.",
    "welch1947": "Welch, B. L. (1947). The generalization of 'Student's' problem when several different population variances are involved. Biometrika, 34(1–2), 28–35.",
    "student1908": "Student. (1908). The probable error of a mean. Biometrika, 6(1), 1–25.",
    "cohen1988": "Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). Lawrence Erlbaum.",
    # non-parametric and categorical
    "mann1947": "Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. The Annals of Mathematical Statistics, 18(1), 50–60.",
    "wilcoxon1945": "Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), 80–83.",
    "kruskal1952": "Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. Journal of the American Statistical Association, 47(260), 583–621.",
    "friedman1937": "Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. Journal of the American Statistical Association, 32(200), 675–701.",
    "pearson1900": "Pearson, K. (1900). On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. Philosophical Magazine, 50(302), 157–175.",
    "yates1934": "Yates, F. (1934). Contingency tables involving small numbers and the χ² test. Supplement to the Journal of the Royal Statistical Society, 1(2), 217–235.",
    # correlation and regression
    "spearman1904": "Spearman, C. (1904). The proof and measurement of association between two things. The American Journal of Psychology, 15(1), 72–101.",
    "wold2001": "Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: A basic tool of chemometrics. Chemometrics and Intelligent Laboratory Systems, 58(2), 109–130.",
    "chong2005": "Chong, I.-G., & Jun, C.-H. (2005). Performance of some variable selection methods when multicollinearity is present. Chemometrics and Intelligent Laboratory Systems, 78(1–2), 103–112.",
    # multivariate
    "jolliffe2002": "Jolliffe, I. T. (2002). Principal component analysis (2nd ed.). Springer.",
    "kaiser1958": "Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis. Psychometrika, 23(3), 187–200.",
    "horn1965": "Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. Psychometrika, 30(2), 179–185.",
    "ward1963": "Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. Journal of the American Statistical Association, 58(301), 236–244.",
    "macqueen1967": "MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability (Vol. 1, pp. 281–297). University of California Press.",
    "rousseeuw1987": "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53–65.",
    "sokal1962": "Sokal, R. R., & Rohlf, F. J. (1962). The comparison of dendrograms by objective methods. Taxon, 11(2), 33–40.",
    "mahalanobis1936": "Mahalanobis, P. C. (1936). On the generalized distance in statistics. Proceedings of the National Institute of Sciences of India, 2(1), 49–55.",
    "rao1952": "Rao, C. R. (1952). Advanced statistical methods in biometric research. Wiley.",
    "singh1981": "Singh, D. (1981). The relative importance of characters affecting genetic divergence. Indian Journal of Genetics and Plant Breeding, 41(2), 237–245.",
    # genotype × environment
    "gauch1992": "Gauch, H. G. (1992). Statistical analysis of regional yield trials: AMMI analysis of factorial designs. Elsevier.",
    "yan2000": "Yan, W., Hunt, L. A., Sheng, Q., & Szlavnics, Z. (2000). Cultivar evaluation and mega-environment investigation based on the GGE biplot. Crop Science, 40(3), 597–605.",
    "eberhart1966": "Eberhart, S. A., & Russell, W. A. (1966). Stability parameters for comparing varieties. Crop Science, 6(1), 36–40.",
    "wricke1962": "Wricke, G. (1962). Über eine Methode zur Erfassung der ökologischen Streubreite in Feldversuchen. Zeitschrift für Pflanzenzüchtung, 47, 92–96.",
    "shukla1972": "Shukla, G. K. (1972). Some statistical aspects of partitioning genotype-environmental components of variability. Heredity, 29(2), 237–245.",
    "purchase2000": "Purchase, J. L., Hatting, H., & van Deventer, C. S. (2000). Genotype × environment interaction of winter wheat (Triticum aestivum L.) in South Africa: II. Stability analysis of yield performance. South African Journal of Plant and Soil, 17(3), 101–107.",
    # trends and time series
    "mann1945": "Mann, H. B. (1945). Nonparametric tests against trend. Econometrica, 13(3), 245–259.",
    "kendall1975": "Kendall, M. G. (1975). Rank correlation methods (4th ed.). Charles Griffin.",
    "hamed1998": "Hamed, K. H., & Rao, A. R. (1998). A modified Mann-Kendall trend test for autocorrelated data. Journal of Hydrology, 204(1–4), 182–196.",
    "hirsch1982": "Hirsch, R. M., Slack, J. R., & Smith, R. A. (1982). Techniques of trend analysis for monthly water quality data. Water Resources Research, 18(1), 107–121.",
    "sen1968": "Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. Journal of the American Statistical Association, 63(324), 1379–1389.",
    "pettitt1979": "Pettitt, A. N. (1979). A non-parametric approach to the change-point problem. Journal of the Royal Statistical Society: Series C (Applied Statistics), 28(2), 126–135.",
    "alexandersson1986": "Alexandersson, H. (1986). A homogeneity test applied to precipitation data. Journal of Climatology, 6(6), 661–675.",
    "buishand1982": "Buishand, T. A. (1982). Some methods for testing the homogeneity of rainfall records. Journal of Hydrology, 58(1–2), 11–27.",
    "sen2012": "Şen, Z. (2012). Innovative trend analysis methodology. Journal of Hydrologic Engineering, 17(9), 1042–1046.",
    "hyndman2021": "Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and practice (3rd ed.). OTexts.",
    "cleveland1990": "Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. Journal of Official Statistics, 6(1), 3–73.",
    "dickey1979": "Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. Journal of the American Statistical Association, 74(366), 427–431.",
    "kwiatkowski1992": "Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. Journal of Econometrics, 54(1–3), 159–178.",
    "ljung1978": "Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. Biometrika, 65(2), 297–303.",
    "hyndman2006": "Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4), 679–688.",
    # structural equation modelling
    "bollen1989": "Bollen, K. A. (1989). Structural equations with latent variables. Wiley.",
    "hair2022": "Hair, J. F., Hult, G. T. M., Ringle, C. M., & Sarstedt, M. (2022). A primer on partial least squares structural equation modeling (PLS-SEM) (3rd ed.). Sage.",
    "henseler2015": "Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion for assessing discriminant validity in variance-based structural equation modeling. Journal of the Academy of Marketing Science, 43(1), 115–135.",
    "fornell1981": "Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation models with unobservable variables and measurement error. Journal of Marketing Research, 18(1), 39–50.",
    "hu1999": "Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance structure analysis: Conventional criteria versus new alternatives. Structural Equation Modeling, 6(1), 1–55.",
    "igolkina2020": "Igolkina, A. A., & Meshcheryakov, G. (2020). semopy: A Python package for structural equation modeling. Structural Equation Modeling, 27(6), 952–963.",
    # data preparation
    "tukey1977": "Tukey, J. W. (1977). Exploratory data analysis. Addison-Wesley.",
    "iglewicz1993": "Iglewicz, B., & Hoaglin, D. C. (1993). How to detect and handle outliers. ASQC Quality Press.",
    "rubin1987": "Rubin, D. B. (1987). Multiple imputation for nonresponse in surveys. Wiley.",
    "barnard1999": "Barnard, J., & Rubin, D. B. (1999). Small-sample degrees of freedom with multiple imputation. Biometrika, 86(4), 948–955.",
    "vanbuuren2018": "van Buuren, S. (2018). Flexible imputation of missing data (2nd ed.). CRC Press.",
    "vanbuuren2011": "van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. Journal of Statistical Software, 45(3), 1–67.",
    "li1991": "Li, K. H., Raghunathan, T. E., & Rubin, D. B. (1991). Large-sample significance levels from multiply imputed data using moment-based statistics and an F reference distribution. Journal of the American Statistical Association, 86(416), 1065–1073.",
    "fisher1915": "Fisher, R. A. (1915). Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population. Biometrika, 10(4), 507–521.",
    "fieller1957": "Fieller, E. C., Hartley, H. O., & Pearson, E. S. (1957). Tests for rank correlation coefficients. I. Biometrika, 44(3/4), 470–481.",
    "troyanskaya2001": "Troyanskaya, O., Cantor, M., Sherlock, G., Brown, P., Hastie, T., Tibshirani, R., Botstein, D., & Altman, R. B. (2001). Missing value estimation methods for DNA microarrays. Bioinformatics, 17(6), 520–525.",
    # software
    "virtanen2020": "Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17(3), 261–272.",
    "seabold2010": "Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python. In Proceedings of the 9th Python in Science Conference (pp. 92–96).",
    "mckinney2010": "McKinney, W. (2010). Data structures for statistical computing in Python. In Proceedings of the 9th Python in Science Conference (pp. 56–61).",
    "pedregosa2011": "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
}

SOFTWARE_KEYS = ["mckinney2010", "virtanen2020", "seabold2010"]

ANOVA_DESIGNS = {
    "oneway": "a one-factor completely randomized design (CRD)",
    "rbd_oneway": "a one-factor randomized complete block design (RBD)",
    "lsd": "a Latin square design",
    "twoway": "a two-factor completely randomized design with interaction",
    "rbd_twoway": "a two-factor randomized complete block design",
    "splitplot": "a split-plot design (main-plot factor tested against the main-plot error, sub-plot factor and interaction against the sub-plot error)",
    "crd_multifactor": "a multi-factor completely randomized design (Type III sums of squares)",
    "rbd_multifactor": "a multi-factor randomized complete block design (Type III sums of squares)",
}

POSTHOC = {
    "tukey": ("Tukey's honestly significant difference test with the Tukey-Kramer adjustment for unequal replication", ["tukey1949", "kramer1956"]),
    "lsd": ("Fisher's least significant difference (critical difference)", ["gomez1984"]),
    "duncan": ("Duncan's multiple range test", ["duncan1955"]),
    "bonferroni": ("pairwise t-tests with a Bonferroni correction", []),
    "games_howell": ("the Games-Howell procedure, which does not assume equal variances", ["games1976"]),
}


def _names(values):
    values = [str(v) for v in (values or []) if v not in (None, "")]
    if not values:
        return "the selected variables"
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + " and " + values[-1]


def _get(mapping, *keys, default=None):
    for key in keys:
        if not isinstance(mapping, dict):
            return default
        mapping = mapping.get(key)
    return default if mapping is None else mapping


def _descriptive(p, r):
    group = p.get("group_var")
    text = (f"Descriptive statistics (mean, median, standard deviation, standard error, coefficient of variation, range, "
            f"skewness and kurtosis) were computed for {_names(p.get('variables'))}"
            + (f", separately for each level of {group}" if group else "")
            + ". Normality was assessed with the Shapiro-Wilk test.")
    return text, ["shapiro1965"]


def _correlation(p, r):
    method = p.get("method", "pearson")
    if method == "spearman":
        text = "Spearman's rank correlation coefficients"
        keys = ["spearman1904"]
    else:
        text = "Pearson product-moment correlation coefficients"
        keys = []
    return (f"{text} were computed between {_names(p.get('variables'))}, each tested against zero with a two-tailed "
            f"test on pairwise-complete observations."), keys


def _parametric(p, r):
    test = p.get("test") or ""
    used = _get(r, "test_used") or ""
    variables = p.get("variables") or []
    keys = ["shapiro1965", "cohen1988"]
    if test == "paired_t":
        body = f"{_names(variables)} were compared with a paired t-test"
        keys.append("student1908")
    elif test == "one_sample_t":
        body = f"The mean of {_names(variables[:1])} was compared with a hypothesised value by a one-sample t-test"
        keys.append("student1908")
    elif test == "z_test":
        body = f"{_names(variables)} were compared with a z-test"
    else:
        if "Welch" in used:
            body = f"{_names(variables)} were compared with Welch's t-test, as Levene's test indicated unequal variances"
            keys += ["welch1947", "levene1960"]
        else:
            body = f"{_names(variables)} were compared with Student's independent-samples t-test after Levene's test for equal variances"
            keys += ["student1908", "levene1960"]
    return (body + ". Normality was checked with the Shapiro-Wilk test, and effect sizes are reported as Cohen's d "
            "and Hedges' g."), keys


def _nonparametric(p, r):
    test = p.get("test") or ""
    variables = _names(p.get("variables"))
    catalogue = {
        "mann_whitney": (f"{variables} were compared with the Mann-Whitney U test, with the rank-biserial correlation as effect size", ["mann1947"]),
        "wilcoxon": (f"{variables} were compared with the Wilcoxon signed-rank test for paired observations, with the "
                     f"matched-pairs rank-biserial correlation as effect size", ["wilcoxon1945"]),
        "kruskal": (f"{variables} were compared with the Kruskal-Wallis H test, a rank-based alternative to one-way "
                    f"analysis of variance that does not assume normally distributed data", ["kruskal1952"]),
        "kruskal_wallis": (f"{variables} were compared with the Kruskal-Wallis H test, a rank-based alternative to "
                           f"one-way analysis of variance that does not assume normally distributed data", ["kruskal1952"]),
        "friedman": (f"{variables} were compared with the Friedman test, a rank-based test for the same units measured "
                     f"under several conditions", ["friedman1937"]),
        "chi_square": (f"The association between {variables} was tested with Pearson's chi-square test of independence "
                       f"(with Yates' continuity correction for 2 × 2 tables)", ["pearson1900", "yates1934"]),
    }
    name = _get(r, "statistics", "test_name") or ""
    text, keys = catalogue.get(test, (f"{variables} were compared with the {name or 'selected non-parametric test'}", []))
    return text + ".", keys


def _anova(p, r):
    design = ANOVA_DESIGNS.get(p.get("design"), "the selected experimental design")
    factors = [f for f in (p.get("factor_a"), p.get("factor_b")) if f]
    block = p.get("block")
    posthoc, posthoc_keys = POSTHOC.get(p.get("posthoc") or "tukey", POSTHOC["tukey"])
    text = (f"{p.get('dependent') or 'The response'} was analysed by analysis of variance for {design}"
            + (f" with {_names(factors)} as treatment factor{'s' if len(factors) > 1 else ''}" if factors else "")
            + (f" and {block} as replication/block" if block else "")
            + ". The standard error of the difference, critical difference (CD) and coefficient of variation were "
            f"calculated, and treatment means were compared by {posthoc}, summarised as compact letter displays. "
            "Residual normality and homogeneity of variances were checked with the Shapiro-Wilk and Levene tests.")
    return text, ["gomez1984", *posthoc_keys, "piepho2004", "shapiro1965", "levene1960"]


def _regression(p, r):
    kind = p.get("type") or "simple"
    dependent = _names(p.get("dependent"))
    independent = _names(p.get("independent"))
    if kind == "plsr":
        components = p.get("components") or 2
        return (f"{dependent} was modelled from {independent} by partial least squares regression with {components} "
                f"latent component{'s' if components != 1 else ''}; the importance of each predictor is summarised by "
                f"variable importance in projection (VIP) scores."), ["wold2001", "chong2005"]
    label = "simple" if kind == "simple" else "multiple"
    return (f"{dependent} was modelled from {independent} by {label} ordinary least squares regression. Coefficients are "
            f"reported with standard errors, t-tests and 95% confidence intervals, with R², adjusted R² and residual "
            f"diagnostics."), []


def _pca(p, r):
    scale = p.get("scale", True)
    varimax = p.get("varimax")
    text = (f"Principal component analysis was performed on {_names(p.get('variables'))} using the "
            f"{'correlation' if scale else 'covariance'} matrix{' (variables standardised to unit variance)' if scale else ''}. "
            "The number of components to retain was judged from the eigenvalues, the scree plot and parallel analysis.")
    keys = ["jolliffe2002", "horn1965"]
    if varimax and varimax != "auto":
        text += f" The first {varimax} components were rotated by the varimax criterion."
        keys.append("kaiser1958")
    return text, keys


def _clustering(p, r):
    method = p.get("method", "kmeans")
    variables = _names(p.get("variables"))
    if method == "hierarchical":
        linkage = p.get("linkage", "ward")
        distance = p.get("distance", "euclidean")
        text = (f"Observations were grouped by agglomerative hierarchical clustering of {variables} using {linkage} "
                f"linkage on {distance} distances; the agreement between the dendrogram and the original distances was "
                f"measured by the cophenetic correlation, and cluster quality by silhouette widths.")
        keys = ["sokal1962", "rousseeuw1987"] + (["ward1963"] if linkage == "ward" else [])
    else:
        text = (f"Observations were grouped into {p.get('k', 3)} clusters by k-means clustering of {variables}; cluster "
                f"quality was assessed with silhouette widths.")
        keys = ["macqueen1967", "rousseeuw1987"]
    return text, keys


def _trend(p, r):
    season = p.get("season")
    alpha = p.get("alpha", 0.05)
    text = (f"Monotonic trends in {_names(p.get('variables'))} were tested with the Mann-Kendall test and its "
            f"modification for serially correlated data"
            + (f", and with the seasonal Mann-Kendall test across {season}" if season else "")
            + ". Trend magnitude was estimated by Sen's slope with its confidence interval. Change points and "
            "homogeneity were examined with the Pettitt, standard normal homogeneity (SNHT), Buishand range and von "
            f"Neumann ratio tests, and trends were also visualised by innovative trend analysis. Tests used α = {alpha}.")
    keys = ["mann1945", "kendall1975", "hamed1998", "sen1968", "pettitt1979", "alexandersson1986", "buishand1982", "sen2012"]
    if season:
        keys.append("hirsch1982")
    return text, keys


def _gxe(p, r):
    replicate = p.get("replicate")
    text = (f"Genotype × environment interaction for {p.get('response') or 'the response'} was analysed with "
            f"{p.get('genotype') or 'genotypes'} tested across {p.get('environment') or 'environments'}"
            + (f" (replicates in {replicate})" if replicate else "")
            + ". The additive main effects and multiplicative interaction (AMMI) model partitioned the interaction into "
            "principal components, and the AMMI stability value was calculated. Stability was also described by the "
            "Eberhart-Russell regression, Wricke's ecovalence and Shukla's stability variance. Genotype and "
            "genotype-by-environment effects were displayed in GGE biplots, including which-won-where and "
            "mean-versus-stability views.")
    return text, ["gauch1992", "purchase2000", "eberhart1966", "wricke1962", "shukla1972", "yan2000"]


def _divergence(p, r):
    return (f"Genetic divergence among {p.get('genotype') or 'genotypes'} for {_names(p.get('traits'))} was assessed by "
            f"Mahalanobis' D² statistic, after testing the differences among genotypes jointly with Wilks' Λ. Genotypes "
            f"were grouped by Tocher's method, average intra- and inter-cluster distances were calculated, and the "
            f"relative contribution of each character to divergence was determined; canonical variates summarised the "
            f"separation."), ["mahalanobis1936", "rao1952", "singh1981"]


def _timeseries(p, r):
    season = p.get("season_length")
    horizon = p.get("horizon")
    model = _get(r, "forecast", "model")
    text = (f"The series {p.get('value') or ''} was examined for stationarity with the augmented Dickey-Fuller and KPSS "
            f"tests and for autocorrelation with the ACF, PACF and Ljung-Box test"
            + (f", and decomposed into trend and seasonal components by STL (season length {season})" if season else "")
            + ". Naive, seasonal naive and drift baselines, exponential smoothing (ETS) and seasonal ARIMA models were "
            "compared on a hold-out period by RMSE, MAE, MAPE and MASE; the best model"
            + (f" ({model})" if model else "")
            + " was refitted to the whole series"
            + (f" to forecast {horizon} periods ahead with prediction intervals." if horizon else " for forecasting."))
    keys = ["hyndman2021", "dickey1979", "kwiatkowski1992", "ljung1978", "hyndman2006"]
    if season:
        keys.append("cleveland1990")
    return text, keys


def _sem(p, r):
    kind = p.get("type") or "pls"
    latents = p.get("latents") or []
    names = _names([latent.get("name") for latent in latents if isinstance(latent, dict)])
    if kind == "cb":
        return (f"A covariance-based structural equation model with the constructs {names} was estimated by maximum "
                f"likelihood. Model fit was judged by χ², CFI, TLI, RMSEA and SRMR, and reliability and convergent "
                f"validity by composite reliability and average variance extracted."), \
            ["bollen1989", "hu1999", "fornell1981", "igolkina2020"]
    return (f"A partial least squares structural equation model with the constructs {names} was estimated with the PLS "
            f"path algorithm, and path significance was assessed by bootstrapping (1,000 resamples). The measurement "
            f"model was evaluated by indicator loadings, Cronbach's α, composite reliability, average variance extracted "
            f"and the heterotrait-monotrait ratio (HTMT)."), ["hair2022", "fornell1981", "henseler2015"]


def _imputation(p, r):
    text, keys = multiple_imputation_paragraph(p.get("m", 20), p.get("max_iter", 10), p.get("seed", 2026))
    text = (f"Missing values in {_names(p.get('columns'))} were handled by multiple imputation by chained equations "
            f"(m = {p.get('m', 20)}, {p.get('max_iter', 10)} iterations, random seed {p.get('seed', 2026)}), using "
            f"{_names(p.get('predictors'))} as predictors and drawing imputations from the posterior predictive "
            f"distribution of Bayesian ridge regressions. Each analysis was repeated on every completed dataset and the "
            f"results were pooled with Rubin's rules and Barnard-Rubin degrees of freedom.")
    analyses = p.get("analyses") or {}
    if analyses.get("group_comparison"):
        spec = analyses["group_comparison"]
        text += (f" The difference in {spec.get('outcome')} between the two levels of {spec.get('group')} was pooled "
                 f"with its Welch standard error.")
    if analyses.get("correlation"):
        spec = analyses["correlation"]
        spearman = spec.get("method") == "spearman"
        between = "between" if len(spec.get("variables") or []) == 2 else "among"
        text += (f" {'Spearman' if spearman else 'Pearson'} correlations {between} {_names(spec.get('variables'))} were pooled "
                 f"on Fisher's z scale{' with the Fieller-Hartley-Pearson variance' if spearman else ''}.")
        keys += ["fisher1915"] + (["fieller1957", "spearman1904"] if spearman else [])
    if analyses.get("anova"):
        spec = analyses["anova"]
        comparison = "Bonferroni-adjusted" if spec.get("comparison", "bonferroni") == "bonferroni" else "unadjusted"
        text += (f" An analysis of variance of {spec.get('response')} with {_names(spec.get('factors'))} as fixed "
                 f"factors{' and ' + spec['block'] + ' as block' if spec.get('block') else ''} was fitted to each "
                 f"completed dataset with sum-to-zero contrasts; each term was tested with the pooled multivariate Wald "
                 f"statistic D1, and level means were compared by {comparison} pooled pairwise t-tests.")
        keys.append("li1991")
    if p.get("regression_dependent"):
        text += f" An ordinary least squares regression of {p['regression_dependent']} was pooled likewise."
    return text, keys


DESCRIBERS = {
    "imputation": _imputation,
    "descriptive": _descriptive,
    "correlation": _correlation,
    "parametric": _parametric,
    "nonparametric": _nonparametric,
    "anova": _anova,
    "regression": _regression,
    "pca": _pca,
    "clustering": _clustering,
    "trend": _trend,
    "gxe": _gxe,
    "divergence": _divergence,
    "timeseries": _timeseries,
    "sem": _sem,
}


def describe_analysis(module, parameters=None, result=None):
    """A methods paragraph and the reference keys for one saved analysis."""
    describer = DESCRIBERS.get(module)
    if describer is None:
        return {"paragraph": f"A {module} analysis was performed.", "references": []}
    text, keys = describer(parameters or {}, result or {})
    return {"paragraph": " ".join(text.split()), "references": [k for k in dict.fromkeys(keys) if k in REFERENCES]}


# ------------------------------------------------------------------------------------------------ recipe steps

OUTLIER_RULES = {
    "iqr": ("values outside 1.5 × the interquartile range from the quartiles (Tukey's fences)", ["tukey1977"]),
    "zscore": ("values more than 3 standard deviations from the mean", []),
    "mad": ("modified z-scores above 3.5, based on the median absolute deviation", ["iglewicz1993"]),
}

OUTLIER_ACTIONS = {
    "flag": "were flagged in an indicator column",
    "to_missing": "were set to missing",
    "winsorize": "were winsorized to the most extreme value inside the fence",
    "remove": "were removed together with their rows",
}


def describe_step(step):
    """One data-preparation step as a sentence (without the data), with any references it needs."""
    kind = step.get("type")
    columns = _names(step.get("columns")) if step.get("columns") else "all applicable columns"
    column = step.get("column") or "the selected column"
    if kind == "drop_missing":
        if step.get("axis") == "columns":
            limit = step.get("max_missing_fraction")
            return (f"Columns with more than {round(100 * limit)}% missing values were removed." if limit is not None
                    else "Columns containing missing values were removed."), []
        return f"Rows with {'any' if step.get('how', 'any') == 'any' else 'all'} missing values in {columns} were removed.", []
    if kind == "fill_missing":
        method = step.get("method", "mean")
        words = {"mean": "the mean", "median": "the median", "mode": "the most common value", "constant": f"the value {step.get('value')}",
                 "interpolate": "linear interpolation", "ffill": "the preceding value", "bfill": "the following value"}.get(method, method)
        group = f" within each level of {step['group_by']}" if step.get("group_by") else ""
        return f"Missing values in {columns} were replaced by {words}{group}.", []
    if kind == "impute_model":
        if step.get("method") == "knn":
            return (f"Missing values in {columns} were imputed from the {step.get('n_neighbors', 5)} nearest complete "
                    f"observations (k-nearest neighbours on standardised variables)."), ["troyanskaya2001", "pedregosa2011"]
        return (f"Missing values in {columns} were imputed once by chained equations with Bayesian ridge regression "
                f"(random seed {step.get('seed', 2026)})."), ["vanbuuren2011", "pedregosa2011"]
    if kind == "drop_duplicates":
        return "Duplicate rows were removed" + (f" (matched on {columns})." if step.get("columns") else "."), []
    if kind == "filter_rows":
        words = {"eq": "equal to", "ne": "not equal to", "lt": "below", "le": "at most", "gt": "above", "ge": "at least",
                 "between": "between", "in": "one of", "not_in": "not one of", "contains": "containing",
                 "is_missing": "missing", "not_missing": "present"}.get(step.get("op"), step.get("op"))
        value = step.get("value")
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        shown = "" if step.get("op") in ("is_missing", "not_missing") else f" {value}" + (f" and {step.get('value2')}" if step.get("op") == "between" else "")
        verb = "retained" if step.get("action", "keep") == "keep" else "excluded"
        return f"Rows with {column} {words}{shown} were {verb}.", []
    if kind == "select_columns":
        return (f"Only {columns} were retained." if step.get("action") == "keep" else f"The columns {columns} were removed."), []
    if kind == "rename_columns":
        pairs = ", ".join(f"{old} → {new}" for old, new in (step.get("mapping") or {}).items())
        return f"Columns were renamed ({pairs}).", []
    if kind == "sort_rows":
        return f"Rows were sorted by {columns}.", []
    if kind == "cast_type":
        return f"{columns} were converted to {step.get('to', 'numeric')} values.", []
    if kind == "clean_text":
        return f"Text labels in {columns} were standardised (spacing, capitalisation and text codes for missing values).", []
    if kind == "recode_values":
        count = len(step.get("mapping") or {})
        target = f" into {step['new_column']}" if step.get("new_column") else ""
        return f"Values of {column} were recoded{target} ({count} rule{'s' if count != 1 else ''}).", []
    if kind == "compute_column":
        return f"A derived variable {step.get('name')} was calculated as {step.get('formula')}.", []
    if kind == "bin_numeric":
        method = step.get("method", "equal_width")
        how = {"equal_width": f"{step.get('bins', 3)} equal-width classes", "quantile": f"{step.get('bins', 4)} quantile classes",
               "custom": "classes with the limits " + ", ".join(map(str, step.get("edges") or []))}.get(method, method)
        return f"{column} was grouped into {how}.", []
    if kind == "handle_outliers":
        rule, keys = OUTLIER_RULES.get(step.get("method", "iqr"), OUTLIER_RULES["iqr"])
        group = f" within each level of {step['group_by']}" if step.get("group_by") else ""
        action = OUTLIER_ACTIONS.get(step.get("action", "flag"), "were handled")
        return f"Outliers in {columns}, defined as {rule}{group}, {action}.", keys
    if kind == "to_long":
        return (f"The table was reshaped from wide to long format, keeping {_names(step.get('id_columns'))} as "
                f"identifiers."), []
    if kind == "to_wide":
        return (f"The table was reshaped from long to wide format, one column per level of {step.get('columns_from') or 'the chosen column'} "
                f"({step.get('aggfunc', 'mean')} of repeated values)."), []
    if kind == "merge":
        return f"A second table was joined on {_names(step.get('on'))} ({step.get('how', 'left')} join).", []
    return f"A {kind} step was applied.", []


def describe_recipe(steps):
    """Numbered sentences for a recipe and the references they need."""
    sentences, keys = [], []
    for step in steps or []:
        sentence, step_keys = describe_step(step)
        sentences.append(sentence)
        keys.extend(step_keys)
    return sentences, [k for k in dict.fromkeys(keys) if k in REFERENCES]


def multiple_imputation_paragraph(m, max_iter, seed):
    return ((f"Missing values were handled by multiple imputation by chained equations (m = {m}, {max_iter} iterations, "
             f"random seed {seed}), drawing imputations from the posterior predictive distribution of Bayesian ridge "
             f"regressions; estimates from the completed datasets were pooled with Rubin's rules using the Barnard-Rubin "
             f"small-sample degrees of freedom."), ["rubin1987", "barnard1999", "vanbuuren2018", "pedregosa2011"])


def software_paragraph():
    return ("Analyses were carried out in StatSathi, which uses Python with pandas, SciPy and statsmodels for data "
            "handling and statistical computation."), list(SOFTWARE_KEYS)


def reference_list(keys):
    """Unique references sorted alphabetically, as they appear in a reference list."""
    return sorted({REFERENCES[k] for k in keys if k in REFERENCES}, key=lambda text: text.lower())
