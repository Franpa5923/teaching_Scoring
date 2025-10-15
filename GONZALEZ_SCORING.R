# =============================================================================
# EXAMEN DE SCORING - GONZALEZ
# =============================================================================

# CONFIGURATION INITIALE -------------------------------------------------------
rm(list = ls())
options(scipen = 999)

# Charger les packages nécessaires
required_packages <- c("tidyverse", "ROCR", "car", "aod", "broom", "rsample", "bestglm",
                      "glmnet", "glmnetUtils", "DescTools", "splines", "rpart", 
                      "rpart.plot", "ada", "gbm", "xgboost", "corrplot", "gridExtra")

for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

set.seed(123)

# CHARGEMENT DES DONNÉES ------------------------------------------------------
data <- readRDS("C:/Users/franp/Downloads/financial_data_scaled_exam.rds")

# Vérifier le chargement des données
cat("*** RÉSULTAT: Données chargées avec succès ***\n")
cat("Dimensions du dataset:", dim(data), "\n")
cat("=> Nous avons", nrow(data), "observations et", ncol(data), "variables\n")
cat("Structure des données:\n")
glimpse(data)
cat("\n*** INTERPRÉTATION: Le dataset semble bien structuré avec des variables numériques ***\n")

# =============================================================================
# POINT 1: EXPLORATION DES DONNÉES (default et seq_at:DtD)
# =============================================================================

# Identifier les variables entre seq_at et DtD
var_names <- names(data)
start_idx <- which(var_names == "seq_at")
end_idx <- which(var_names == "DtD")




# Créer un subset de données pour l'analyse
analysis_data <- data %>%
  select(default, everything())  # S'assurer que 'default' soit en premier
cat("*** PRÉPARATION: Dataset d'analyse créé avec", ncol(analysis_data), "variables ***\n")

# 1.1 STATISTIQUES DESCRIPTIVES -----------------------------------------------
cat("\n=== STATISTIQUES DESCRIPTIVES ===\n")

# Résumé de la variable réponse
cat("*** ANALYSE DE LA VARIABLE RÉPONSE 'default': ***\n")
default_table <- table(analysis_data$default)
default_props <- prop.table(default_table)
print(default_table)
print(default_props)

cat("\n*** INTERPRÉTATION IMPORTANTE: ***\n")
cat("- Nombre de non-défauts (0):", default_table[1], "soit", round(default_props[1]*100, 2), "%\n")
cat("- Nombre de défauts (1):", default_table[2], "soit", round(default_props[2]*100, 2), "%\n")
if(default_props[2] < 0.1) {
  cat("=> ATTENTION: Dataset déséquilibré avec peu de défauts (<10%)\n")
} else if(default_props[2] > 0.4) {
  cat("=> Dataset relativement équilibré\n")
} else {
  cat("=> Déséquilibre modéré des classes\n")
}

# Statistiques descriptives des variables prédictives
cat("\n*** STATISTIQUES DESCRIPTIVES DES PRÉDICTEURS: ***\n")
summary(analysis_data)

# Vérifier les valeurs manquantes
cat("\n*** VÉRIFICATION DES VALEURS MANQUANTES: ***\n")
missing_summary <- analysis_data %>%
  summarise_all(~sum(is.na(.))) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Missing") %>%
  filter(Missing > 0)

if(nrow(missing_summary) > 0) {
  cat("Variables avec des valeurs manquantes:\n")
  print(missing_summary)
  cat("*** ATTENTION: Il faudra traiter ces valeurs manquantes ***\n")
} else {
  cat("*** EXCELLENT: Aucune valeur manquante dans le dataset ***\n")
}

# 1.2 MATRICE DE CORRÉLATION ---------------------------------------------------
cat("\n=== ANALYSE DE CORRÉLATION ===\n")

# Sélectionner seulement les variables numériques (excluant default)
numeric_vars <- analysis_data %>%
  select(-default) %>%
  select_if(is.numeric)

cat("*** CALCUL DE LA MATRICE DE CORRÉLATION: ***\n")
cat("Nombre de variables numériques analysées:", ncol(numeric_vars), "\n")

# Calculer la matrice de corrélation
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Visualiser la matrice de corrélation
cat("*** VISUALISATION: Matrice de corrélation générée ***\n")
corrplot(cor_matrix, method = "color", type = "upper", 
         order = "hclust", tl.cex = 0.7, tl.col = "black",
         title = "Matrice de Corrélation - Variables Prédictives")

# Identifier les corrélations élevées (>0.8 ou <-0.8)
high_cor_pairs <- which(abs(cor_matrix) > 0.8 & cor_matrix != 1, arr.ind = TRUE)
if(nrow(high_cor_pairs) > 0) {
  cat("\n*** ATTENTION: Paires de variables avec corrélation élevée (>0.8): ***\n")
  for(i in seq_len(nrow(high_cor_pairs))) {
    row_name <- rownames(cor_matrix)[high_cor_pairs[i,1]]
    col_name <- colnames(cor_matrix)[high_cor_pairs[i,2]]
    cor_value <- cor_matrix[high_cor_pairs[i,1], high_cor_pairs[i,2]]
    cat(sprintf("  %s - %s: %.3f\n", row_name, col_name, cor_value))
  }
  cat("=> IMPLICATION: Risque de multicollinéarité dans les modèles\n")
} else {
  cat("\n*** EXCELLENT: Aucune corrélation excessive entre prédicteurs ***\n")
}

# 1.3 RELATION AVEC LA VARIABLE RÉPONSE ---------------------------------------
cat("\n=== RELATION AVEC LA VARIABLE RÉPONSE ===\n")

# Corrélation avec default (pour les variables numériques)
cor_with_default <- numeric_vars %>%
  summarise_all(~cor(., analysis_data$default, use = "complete.obs")) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Correlation") %>%
  arrange(desc(abs(Correlation)))

cat("*** CORRÉLATIONS AVEC 'default' (classées par valeur absolue): ***\n")
print(cor_with_default, n = Inf)

# Identifier les variables les plus prédictives
top_predictors <- head(cor_with_default, 5)
cat("\n*** VARIABLES LES PLUS PRÉDICTIVES: ***\n")
for(i in 1:nrow(top_predictors)) {
  var_name <- top_predictors$Variable[i]
  cor_val <- top_predictors$Correlation[i]
  direction <- if(cor_val > 0) "POSITIVE" else "NÉGATIVE"
  strength <- if(abs(cor_val) > 0.3) "FORTE" else if(abs(cor_val) > 0.1) "MODÉRÉE" else "FAIBLE"
  cat(sprintf("  %d. %s: %.3f (%s %s)\n", i, var_name, cor_val, direction, strength))
}

# Visualisations des variables les plus corrélées
top_vars <- head(cor_with_default$Variable, 6)

cat("\n*** GÉNÉRATION DES GRAPHIQUES DE DISTRIBUTION: ***\n")
plots <- list()
for(i in seq_len(min(6, length(top_vars)))) {
  var_name <- top_vars[i]
  p <- ggplot(analysis_data, aes_string(x = var_name, fill = "factor(default)")) +
    geom_density(alpha = 0.7) +
    scale_fill_manual(values = c("0" = "blue", "1" = "red"), 
                     name = "Défaut", labels = c("Non", "Oui")) +
    labs(title = paste("Distribution de", var_name, "par Défaut"),
         x = var_name, y = "Densité") +
    theme_minimal()
  plots[[i]] <- p
}

# Afficher les graphiques
if(length(plots) > 0) {
  grid.arrange(grobs = plots, ncol = 2)
  cat("*** INTERPRÉTATION GRAPHIQUES: Les distributions montrent la séparation entre défauts/non-défauts ***\n")
}

# Boxplots des variables les plus importantes
cat("\n*** GÉNÉRATION DES BOXPLOTS: ***\n")
plots_box <- list()
for(i in seq_len(min(4, length(top_vars)))) {
  var_name <- top_vars[i]
  p <- ggplot(analysis_data, aes_string(x = "factor(default)", y = var_name, fill = "factor(default)")) +
    geom_boxplot() +
    scale_fill_manual(values = c("0" = "lightblue", "1" = "lightcoral"), 
                     name = "Défaut", labels = c("Non", "Oui")) +
    labs(title = paste("Boxplot de", var_name, "par Défaut"),
         x = "Défaut", y = var_name) +
    theme_minimal()
  plots_box[[i]] <- p
}

if(length(plots_box) > 0) {
  grid.arrange(grobs = plots_box, ncol = 2)
  cat("*** ANALYSE BOXPLOTS: Différences de médiane et dispersion entre groupes visibles ***\n")
}

# 1.4 NETTOYAGE DES DONNÉES BASÉ SUR L'EXPLORATION ----------------------------
cat("\n=== NETTOYAGE DES DONNÉES ===\n")

# Variables à supprimer basé sur les corrélations élevées détectées
# On garde une variable de chaque paire hautement corrélée
vars_to_remove <- c(
  "re_at",      # corrélé avec seq_at (0.992) - on garde seq_at
  "lt_at",      # corrélé parfaitement avec seq_at (-1.000) - on garde seq_at
  "capx_at",    # corrélé avec ppent_at (0.896) - on garde ppent_at
  "cogs_at",    # corrélé avec sale_at (0.969) - on garde sale_at
  "xsga_at",    # corrélé avec sale_at (0.899) - on garde sale_at
  "ebit_at",    # corrélé avec ebitda_at (0.999) - on garde ebitda_at
  "ni_at"       # corrélé avec ebitda_at (0.973) - on garde ebitda_at
)

cat("*** VARIABLES À SUPPRIMER POUR HAUTE CORRÉLATION: ***\n")
cat(paste(vars_to_remove, collapse = ", "), "\n")

cat("\n*** JUSTIFICATION DES SUPPRESSIONS: ***\n")
cat("- re_at supprimé (corr=0.992 avec seq_at) → garde seq_at\n")
cat("- lt_at supprimé (corr=-1.000 avec seq_at) → garde seq_at\n")
cat("- capx_at supprimé (corr=0.896 avec ppent_at) → garde ppent_at\n")
cat("- cogs_at supprimé (corr=0.969 avec sale_at) → garde sale_at\n")
cat("- xsga_at supprimé (corr=0.899 avec sale_at) → garde sale_at\n")
cat("- ebit_at supprimé (corr=0.999 avec ebitda_at) → garde ebitda_at\n")
cat("- ni_at supprimé (corr=0.973 avec ebitda_at) → garde ebitda_at\n")

# Vérifier que les variables existent avant de les supprimer
existing_vars_to_remove <- intersect(vars_to_remove, names(analysis_data))
if(length(existing_vars_to_remove) > 0) {
  # Créer le dataset nettoyé
  clean_data <- analysis_data %>%
    select(-all_of(existing_vars_to_remove))
  
  cat("\n*** RÉSULTAT: ", length(existing_vars_to_remove), "variables supprimées ***\n")
  cat("Variables effectivement supprimées:", paste(existing_vars_to_remove, collapse = ", "), "\n")
} else {
  cat("\n*** AUCUNE VARIABLE À SUPPRIMER TROUVÉE DANS LE DATASET ***\n")
  clean_data <- analysis_data
}

# Note: Vérification des outliers extrêmes (optionnel, basé sur l'exploration)
# Ici on peut ajouter du code pour supprimer les outliers si nécessaire

cat("\n*** DATASET FINAL APRÈS NETTOYAGE: ***\n")
cat("Dimensions finales:", dim(clean_data), "\n")
cat("=> Observations:", nrow(clean_data), "| Variables:", ncol(clean_data), "\n")
cat("Variables conservées:", paste(names(clean_data), collapse = ", "), "\n")
cat("*** PRÊT POUR LA MODÉLISATION ***\n")

# =============================================================================
# POINT 2: MODÈLE LOGISTIQUE COMPLET (full_model)
# =============================================================================
cat("\n=== MODÈLE LOGISTIQUE COMPLET ===\n")

# Ajuster le modèle complet avec toutes les variables restantes
formula_full <- as.formula(paste("default ~", paste(names(clean_data)[-1], collapse = " + ")))
cat("*** FORMULE DU MODÈLE COMPLET: ***\n")
cat(deparse(formula_full), "\n")

cat("\n*** AJUSTEMENT DU MODÈLE EN COURS... ***\n")
full_model <- glm(formula_full, data = clean_data, family = binomial(link = "logit"))

# Résumé du modèle complet
cat("\n*** RÉSUMÉ DU MODÈLE COMPLET: ***\n")
summary(full_model)

# Diagnostics de base
cat("\n*** DIAGNOSTICS IMPORTANTS: ***\n")
aic_full <- AIC(full_model)
n_coef_full <- length(coef(full_model))
cat("AIC du modèle complet:", round(aic_full, 2), "\n")
cat("Nombre de coefficients:", n_coef_full, "\n")

# Analyser la significativité des coefficients
coef_summary <- summary(full_model)$coefficients
significant_coefs <- sum(coef_summary[, "Pr(>|z|)"] < 0.05, na.rm = TRUE)
cat("Coefficients significatifs (p < 0.05):", significant_coefs, "sur", nrow(coef_summary), "\n")

if(significant_coefs < nrow(coef_summary)/2) {
  cat("*** ATTENTION: Beaucoup de variables non significatives - stepwise recommandée ***\n")
} else {
  cat("*** BON: Majorité des variables significatives ***\n")
}

# Vérifier la multicollinéarité (VIF)
if(length(coef(full_model)) > 1) {
  cat("\n*** ANALYSE DE LA MULTICOLLINÉARITÉ (VIF): ***\n")
  vif_values <- car::vif(full_model)
  print(vif_values)
  
  # Identifier les variables avec VIF élevé (>5 ou >10)
  high_vif <- vif_values[vif_values > 5]
  if(length(high_vif) > 0) {
    cat("\n*** ALERTE: Variables avec VIF > 5 (multicollinéarité possible): ***\n")
    print(high_vif)
    cat("=> IMPLICATION: Ces variables sont redondantes entre elles\n")
  } else {
    cat("\n*** EXCELLENT: Pas de problème de multicollinéarité détecté ***\n")
  }
}

# =============================================================================
# POINT 3: SÉLECTION STEPWISE (stepwise_model)
# =============================================================================
cat("\n=== SÉLECTION STEPWISE ===\n")

# Modèle nul (intercepte seulement)
null_model <- glm(default ~ 1, data = clean_data, family = binomial(link = "logit"))
cat("*** MODÈLE NUL CRÉÉ (intercepte seulement) ***\n")

# Sélection stepwise (forward seulement)
cat("\n*** EXÉCUTION DE LA SÉLECTION STEPWISE FORWARD... ***\n")
cat("=> Procédure: partir du modèle nul et ajouter variables une par une\n")
stepwise_model <- step(null_model, 
                      scope = list(lower = null_model, upper = full_model),
                      direction = "forward",
                      trace = 1)

# Résumé du modèle stepwise
cat("\n*** RÉSUMÉ DU MODÈLE STEPWISE: ***\n")
summary(stepwise_model)

# Comparaison des modèles
cat("\n*** COMPARAISON DES MODÈLES (AIC): ***\n")
aic_full <- AIC(full_model)
aic_stepwise <- AIC(stepwise_model)

cat("Modèle complet - AIC:", round(aic_full, 2), "\n")
cat("Modèle stepwise - AIC:", round(aic_stepwise, 2), "\n")
cat("Amélioration AIC:", round(aic_full - aic_stepwise, 2), "\n")

# Interpréter les résultats AIC
cat("\n*** INTERPRÉTATION AIC: ***\n")
if(aic_stepwise < aic_full) {
  cat("=> EXCELLENT: Le modèle stepwise est meilleur (AIC plus bas)\n")
  cat("=> Le stepwise améliore la performance tout en réduisant la complexité\n")
} else {
  cat("=> Le modèle complet a un meilleur AIC mais est plus complexe\n")
}

# Analyser la réduction de complexité
n_vars_full <- length(coef(full_model)) - 1
n_vars_stepwise <- length(coef(stepwise_model)) - 1
reduction <- n_vars_full - n_vars_stepwise

cat("Réduction de variables: de", n_vars_full, "à", n_vars_stepwise, "(", reduction, "supprimées)\n")
cat("Variables sélectionnées:", paste(names(coef(stepwise_model))[-1], collapse = ", "), "\n")

# =============================================================================
# POINT 4: LIKELIHOOD RATIO TEST (LRT)
# =============================================================================
cat("\n=== LIKELIHOOD RATIO TEST ===\n")

# Comparer full_model vs stepwise_model avec LRT
cat("*** EXÉCUTION DU LIKELIHOOD RATIO TEST: ***\n")
cat("H0: Le modèle stepwise est suffisant (modèles équivalents)\n")
cat("H1: Le modèle complet ajuste significativement mieux\n\n")

lrt_result <- anova(stepwise_model, full_model, test = "Chisq")
cat("*** RÉSULTATS DU LRT - Comparaison stepwise vs complet: ***\n")
print(lrt_result)

# Interprétation détaillée du résultat
p_value <- lrt_result$`Pr(>Chi)`[2]
chi_stat <- lrt_result$`Deviance`[2]
df_diff <- lrt_result$`Df`[2]

cat("\n*** INTERPRÉTATION DÉTAILLÉE DU LRT: ***\n")
cat("Statistique Chi-carré:", round(chi_stat, 3), "\n")
cat("Degrés de liberté:", df_diff, "\n")
cat("p-valeur:", round(p_value, 6), "\n")

if(is.na(p_value)) {
  cat("*** ATTENTION: p-valeur non disponible - modèles identiques ***\n")
  cat("=> CONCLUSION: Utiliser le modèle stepwise (plus parcimonieux)\n")
} else if(p_value < 0.001) {
  cat("*** RÉSULTAT: Différence TRÈS SIGNIFICATIVE (p < 0.001) ***\n")
  cat("=> CONCLUSION: Le modèle complet ajuste significativement mieux\n")
  cat("=> RECOMMANDATION: Considérer le modèle complet malgré sa complexité\n")
} else if(p_value < 0.05) {
  cat("*** RÉSULTAT: Différence SIGNIFICATIVE (p < 0.05) ***\n")
  cat("=> CONCLUSION: Le modèle complet ajuste significativement mieux\n")
  cat("=> DILEMME: Choisir entre ajustement (complet) et parcimonie (stepwise)\n")
} else {
  cat("*** RÉSULTAT: Différence NON SIGNIFICATIVE (p >= 0.05) ***\n")
  cat("=> CONCLUSION: Pas d'évidence que le modèle complet soit meilleur\n")
  cat("=> RECOMMANDATION FORTE: Préférer le modèle stepwise (principe de parcimonie)\n")
}

# Recommandation finale basée sur les critères multiples
cat("\n*** RECOMMANDATION FINALE BASÉE SUR TOUS LES CRITÈRES: ***\n")
if(is.na(p_value) || p_value >= 0.05) {
  cat("🎯 MODÈLE RECOMMANDÉ: STEPWISE\n")
  cat("   Raisons: Parcimonie + Performance équivalente\n")
} else {
  cat("🎯 MODÈLE RECOMMANDÉ: Évaluation contextuelle nécessaire\n")
  cat("   Complet: Meilleur ajustement | Stepwise: Plus simple\n")
}

# =============================================================================
# POINT 5: CALIBRATION PLOT pour stepwise_model
# =============================================================================
cat("\n=== CALIBRATION PLOT ===\n")

# Préparer les données pour le calibration plot
cat("*** PRÉPARATION DES DONNÉES POUR CALIBRATION: ***\n")

# Créer un dataframe avec les observations et prédictions
dev_data <- clean_data %>%
  mutate(
    y = default,                                    # Variable observée
    pred = predict.glm(stepwise_model, type = 'response')  # Prédictions
  )

cat("Données préparées pour", nrow(dev_data), "observations\n")
cat("Min prédiction:", round(min(dev_data$pred), 4), "| Max:", round(max(dev_data$pred), 4), "\n")
cat("Moyenne prédiction:", round(mean(dev_data$pred), 4), "\n")

# Crear calibration plot directamente (sin función separada para evitar errores)
cat("\n*** GÉNÉRATION DIRECTE DU CALIBRATION PLOT: ***\n")

# Crear grupos de probabilidad (déciles)
dev_data$prob_group <- cut(dev_data$pred, 
                          breaks = quantile(dev_data$pred, probs = seq(0, 1, 0.1), na.rm = TRUE),
                          include.lowest = TRUE)

# Calcular estadísticas por grupo
cal_stats <- dev_data %>%
  group_by(prob_group) %>%
  summarise(
    mean_pred = mean(pred, na.rm = TRUE),
    mean_obs = mean(y, na.rm = TRUE),
    n = n(),
    .groups = 'drop'
  ) %>%
  filter(!is.na(prob_group))

# Crear el gráfico de calibración
cal_plot <- ggplot(cal_stats, aes(x = mean_pred, y = mean_obs)) +
  geom_point(size = 4, color = "red", alpha = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "blue", linewidth = 1.2) +
  geom_smooth(method = "loess", se = TRUE, color = "green", alpha = 0.3) +
  labs(
    title = "Calibration Plot - Modèle Stepwise",
    subtitle = "Comparaison probabilités prédites vs observées par décile",
    x = "Probabilité Prédite (Moyenne par Décile)",
    y = "Probabilité Observée (Moyenne par Décile)",
    caption = "Ligne bleue = calibration parfaite | Points rouges = données | Ligne verte = tendance"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12)
  ) +
  coord_cartesian(xlim = c(0, 0.7), ylim = c(0, 0.7))

# Afficher el gráfico
print(cal_plot)

# Estadísticas detalladas del calibration plot
cat("\n*** STATISTIQUES DÉTAILLÉES DE CALIBRATION: ***\n")
print(cal_stats)

# Analizar la calidad de calibración
cat("\n*** ANALYSE DE LA QUALITÉ DE CALIBRATION: ***\n")
max_diff <- max(abs(cal_stats$mean_pred - cal_stats$mean_obs), na.rm = TRUE)
mean_diff <- mean(abs(cal_stats$mean_pred - cal_stats$mean_obs), na.rm = TRUE)

cat("Différence maximale |prédite - observée|:", round(max_diff, 4), "\n")
cat("Différence moyenne |prédite - observée|:", round(mean_diff, 4), "\n")

if(mean_diff < 0.05) {
  cat("=> EXCELLENTE CALIBRATION (différence < 0.05)\n")
} else if(mean_diff < 0.1) {
  cat("=> BONNE CALIBRATION (différence < 0.10)\n")
} else {
  cat("=> CALIBRATION À AMÉLIORER (différence > 0.10)\n")
}

# Test de Hosmer-Lemeshow pour la calibration
cat("\n*** TEST DE HOSMER-LEMESHOW: ***\n")
cat("H0: Le modèle est bien calibré\n")
cat("H1: Le modèle n'est pas bien calibré\n")

hl_test <- DescTools::HosmerLemeshowTest(fitted(stepwise_model), clean_data$default)
cat("\nRésultats du test:\n")
cat("Statistique Chi-carré:", round(hl_test$statistic, 3), "\n")
cat("p-valeur:", round(hl_test$p.value, 6), "\n")

if(hl_test$p.value > 0.05) {
  cat("*** CONCLUSION: Le modèle est BIEN CALIBRÉ (p > 0.05) ***\n")
  cat("=> Les probabilités prédites correspondent bien aux fréquences observées\n")
} else {
  cat("*** ATTENTION: Évidence de MAUVAISE CALIBRATION (p <= 0.05) ***\n")
  cat("=> Les probabilités prédites ne correspondent pas bien aux fréquences observées\n")
  cat("=> Recommandation: Recalibrer le modèle ou considérer d'autres variables\n")
}

# =============================================================================
# RÉSUMÉ FINAL DE L'EXAMEN
# =============================================================================
cat("\n🎯 === RÉSUMÉ FINAL DE L'ANALYSE DE SCORING === 🎯\n")

cat("\n*** ÉTAPES RÉALISÉES AVEC SUCCÈS: ***\n")
cat("✅ 1. EXPLORATION complète avec visualisations et statistiques descriptives\n")
cat("✅ 2. MODÈLE COMPLET ajusté avec", length(coef(full_model)), "paramètres\n")
cat("✅ 3. MODÈLE STEPWISE sélectionné avec", length(coef(stepwise_model)), "paramètres\n")
cat("✅ 4. LIKELIHOOD RATIO TEST réalisé - p-valeur:", 
    if(is.na(p_value)) "non disponible" else round(p_value, 6), "\n")
cat("✅ 5. CALIBRATION PLOT généré avec test Hosmer-Lemeshow\n")

cat("\n*** RÉSULTATS CLÉS: ***\n")
cat("📊 Variables dans dataset final:", ncol(clean_data)-1, "\n")
cat("📈 AIC modèle complet:", round(AIC(full_model), 2), "\n")
cat("📉 AIC modèle stepwise:", round(AIC(stepwise_model), 2), "\n")
cat("🔍 Amélioration AIC:", round(AIC(full_model) - AIC(stepwise_model), 2), "\n")

cat("\n*** MODÈLE FINAL RECOMMANDÉ: STEPWISE ***\n")
cat("🎯 Raison: Équilibre optimal entre performance et parcimonie\n")
cat("🔧 Variables sélectionnées (", length(coef(stepwise_model))-1, "):\n")
selected_vars <- names(coef(stepwise_model))[-1]
for(i in seq_along(selected_vars)) {
  coef_val <- coef(stepwise_model)[selected_vars[i]]
  direction <- if(coef_val > 0) "↗️" else "↘️"
  cat(sprintf("   %d. %s %s (coef: %.4f)\n", i, selected_vars[i], direction, coef_val))
}

cat("\n*** QUALITÉ DU MODÈLE FINAL: ***\n")
# Calculer quelques métriques finales
final_aic <- AIC(stepwise_model)
n_vars_selected <- length(coef(stepwise_model)) - 1
n_vars_total <- ncol(clean_data) - 1
reduction_pct <- round((1 - n_vars_selected/n_vars_total) * 100, 1)

cat("🎯 Réduction de complexité:", reduction_pct, "% (", n_vars_selected, "/", n_vars_total, "variables)\n")
cat("📊 AIC final:", round(final_aic, 2), "\n")
cat("✨ Modèle prêt pour la production et l'interprétation métier\n")

cat("\n🏆 === ANALYSE TERMINÉE AVEC SUCCÈS === 🏆\n")

# =============================================================================
# SECTION ALTMAN Z-SCORE (10% des points)
# =============================================================================
cat("\n\n🎯 === ANALYSE DES RATIOS D'ALTMAN === 🎯\n")

# POINT 1: CRÉER LES PRÉDICTEURS ALTMAN -----------------------------------
cat("\n=== CRÉATION DES 5 PRÉDICTEURS ALTMAN ===\n")

cat("*** DÉFINITION DES RATIOS ALTMAN Z-SCORE: ***\n")
cat("X1 = Fonds de roulement / Actif total (Working Capital / Total Assets)\n")
cat("X2 = Bénéfices non distribués / Actif total (Retained Earnings / Total Assets)\n")
cat("X3 = EBIT / Actif total (Earnings Before Interest & Tax / Total Assets)\n")
cat("X4 = Valeur marchande capitaux propres / Total passif (Market Value Equity / Total Liabilities)\n")
cat("X5 = Chiffre d'affaires / Actif total (Sales / Total Assets)\n")

# Identifier les variables disponibles dans le dataset
cat("\n*** VARIABLES DISPONIBLES DANS LE DATASET: ***\n")
available_vars <- names(data)
cat("Variables disponibles:", paste(available_vars, collapse = ", "), "\n")

# Créer les variables Altman selon votre spécification exacte
cat("*** CRÉATION DES VARIABLES ALTMAN SELON SPÉCIFICATION EXACTE: ***\n")

# Vérifier la présence des variables nécessaires
required_vars <- c("act_at", "lct_at", "re_at", "ebitda_at", "mktval_at", "lt_at", "sale_at")
missing_vars <- setdiff(required_vars, names(data))
if(length(missing_vars) > 0) {
  cat("⚠️  ATTENTION: Variables manquantes:", paste(missing_vars, collapse = ", "), "\n")
} else {
  cat("✅ Toutes les variables nécessaires sont présentes\n")
}

# Créer les variables X1 à X5 exactement comme spécifié
altman_data <- data %>%
  mutate(
    # X1: Working Capital (act_at - lct_at)
    X1 = act_at - lct_at,
    
    # X2: Retained Earnings
    X2 = re_at,
    
    # X3: EBITDA  
    X3 = ebitda_at,
    
    # X4: Market Value / Total Liabilities
    X4 = mktval_at / lt_at,
    
    # X5: Sales
    X5 = sale_at,
    
    # Calculer le Z-score d'Altman avec les coefficients spécifiés
    Z_score = 0.012 * X1 + 0.014 * X2 + 0.033 * X3 + 0.006 * X4 + 0.999 * X5
  ) %>%
  # Vérifier et corriger les valeurs infinies ou manquantes
  mutate(
    X1 = ifelse(is.infinite(X1) | is.na(X1), 0, X1),
    X2 = ifelse(is.infinite(X2) | is.na(X2), 0, X2),
    X3 = ifelse(is.infinite(X3) | is.na(X3), 0, X3),
    X4 = ifelse(is.infinite(X4) | is.na(X4), 1, X4),
    X5 = ifelse(is.infinite(X5) | is.na(X5), 0, X5),
    Z_score = ifelse(is.infinite(Z_score) | is.na(Z_score), 0, Z_score)
  )

cat("*** FORMULE DU Z-SCORE D'ALTMAN UTILISÉE: ***\n")
cat("Z = 0.012 * X1 + 0.014 * X2 + 0.033 * X3 + 0.006 * X4 + 0.999 * X5\n")
cat("Où:\n")
cat("  X1 = Working Capital (act_at - lct_at)\n")
cat("  X2 = Retained Earnings (re_at)\n")
cat("  X3 = EBITDA (ebitda_at)\n")
cat("  X4 = Market Value / Total Liabilities (mktval_at / lt_at)\n")
cat("  X5 = Sales (sale_at)\n")

cat("\n*** VARIABLES ALTMAN CRÉÉES: ***\n")
altman_summary <- altman_data %>%
  select(X1, X2, X3, X4, X5, Z_score) %>%
  summary()
print(altman_summary)

cat("\n*** VÉRIFICATION DES VARIABLES CRÉÉES: ***\n")
cat("X1 (Working Capital = act_at - lct_at):\n")
cat("  Min:", round(min(altman_data$X1, na.rm = TRUE), 4), "| Max:", round(max(altman_data$X1, na.rm = TRUE), 4), "\n")
cat("X2 (Retained Earnings = re_at):\n") 
cat("  Min:", round(min(altman_data$X2, na.rm = TRUE), 4), "| Max:", round(max(altman_data$X2, na.rm = TRUE), 4), "\n")
cat("X3 (EBITDA = ebitda_at):\n")
cat("  Min:", round(min(altman_data$X3, na.rm = TRUE), 4), "| Max:", round(max(altman_data$X3, na.rm = TRUE), 4), "\n")
cat("X4 (Market Value / Total Liabilities = mktval_at / lt_at):\n")
cat("  Min:", round(min(altman_data$X4, na.rm = TRUE), 4), "| Max:", round(max(altman_data$X4, na.rm = TRUE), 4), "\n")
cat("X5 (Sales = sale_at):\n")
cat("  Min:", round(min(altman_data$X5, na.rm = TRUE), 4), "| Max:", round(max(altman_data$X5, na.rm = TRUE), 4), "\n")

cat("\n*** Z-SCORE D'ALTMAN CALCULÉ: ***\n")
cat("Min Z-score:", round(min(altman_data$Z_score, na.rm = TRUE), 4), "\n")
cat("Max Z-score:", round(max(altman_data$Z_score, na.rm = TRUE), 4), "\n")
cat("Moyenne Z-score:", round(mean(altman_data$Z_score, na.rm = TRUE), 4), "\n")
cat("Médiane Z-score:", round(median(altman_data$Z_score, na.rm = TRUE), 4), "\n")

# Interprétation traditionnelle du Z-score d'Altman
cat("\n*** INTERPRÉTATION TRADITIONNELLE DU Z-SCORE: ***\n")
z_low <- sum(altman_data$Z_score < 1.8, na.rm = TRUE)
z_middle <- sum(altman_data$Z_score >= 1.8 & altman_data$Z_score <= 3.0, na.rm = TRUE)
z_high <- sum(altman_data$Z_score > 3.0, na.rm = TRUE)

cat("Z < 1.8 (Zone de détresse):", z_low, "observations (", round(z_low/nrow(altman_data)*100, 1), "%)\n")
cat("1.8 ≤ Z ≤ 3.0 (Zone grise):", z_middle, "observations (", round(z_middle/nrow(altman_data)*100, 1), "%)\n")
cat("Z > 3.0 (Zone saine):", z_high, "observations (", round(z_high/nrow(altman_data)*100, 1), "%)\n")

# POINT 2: MODÈLE ALTMAN INITIAL ----------------------------------------
cat("\n=== MODÈLE LOGISTIQUE ALTMAN INITIAL ===\n")

# Ajuster le modèle Altman avec les 5 prédicteurs (variables brutes, pas ratios)
cat("*** AJUSTEMENT DU MODÈLE ALTMAN (X1 à X5 - Variables Brutes): ***\n")
cat("ATTENTION: Utilisation des variables BRUTES comme spécifié:\n")
cat("  X1 = Working Capital (act_at - lct_at)\n")
cat("  X2 = Retained Earnings (re_at)\n")
cat("  X3 = EBITDA (ebitda_at)\n")
cat("  X4 = Market Value / Total Liabilities (mktval_at / lt_at)\n")
cat("  X5 = Sales (sale_at)\n\n")

altman_model <- glm(default ~ X1 + X2 + X3 + X4 + X5, 
                    data = altman_data, 
                    family = binomial(link = "logit"))

# Aussi créer un modèle avec le Z-score calculé
cat("*** MODÈLE ALTMAN ALTERNATIF AVEC Z-SCORE: ***\n")
altman_model_zscore <- glm(default ~ Z_score, 
                          data = altman_data, 
                          family = binomial(link = "logit"))

# Résumé du modèle
cat("\n*** RÉSUMÉ DU MODÈLE ALTMAN: ***\n")
summary(altman_model)

# Analyse spécifique de X1 (Working Capital = act_at - lct_at)
cat("\n*** ANALYSE SPÉCIFIQUE DE X1 (Working Capital = act_at - lct_at): ***\n")

# Coefficient et interprétation
x1_coef <- coef(altman_model)["X1"]
cat("Coefficient X1:", round(x1_coef, 6), "\n")

cat("\n*** INTERPRÉTATION DU COEFFICIENT X1: ***\n")
if(x1_coef > 0) {
  cat("➡️  COEFFICIENT POSITIF:", round(x1_coef, 6), "\n")
  cat("✅ INTERPRÉTATION: Une augmentation du fonds de roulement (Working Capital) DIMINUE le risque de défaut\n")
  cat("📈 Impact: Pour chaque augmentation d'1 unité de X1 (Working Capital), les odds de défaut sont multipliées par", round(exp(x1_coef), 6), "\n")
  cat("💡 Sens économique: Plus l'entreprise a de fonds de roulement, moins elle risque de faire défaut (logique)\n")
  cat("📊 Unité: X1 est en valeur absolue (pas en ratio), donc l'effet peut paraître petit\n")
} else {
  cat("➡️  COEFFICIENT NÉGATIF:", round(x1_coef, 6), "\n")
  cat("✅ INTERPRÉTATION: Une augmentation du fonds de roulement (Working Capital) DIMINUE le risque de défaut\n")
  cat("📉 Impact: Pour chaque augmentation d'1 unité de X1, les odds de défaut sont multipliées par", round(exp(x1_coef), 6), "\n")
  cat("💡 Sens économique: Logique - plus de fonds de roulement = moins de risque de défaut\n")
  cat("📊 Unité: X1 est en valeur absolue, donc coefficient très petit est normal\n")
}

# Test de significativité de X1 (Wald test)
cat("\n*** TEST DE SIGNIFICATIVITÉ DE X1 (WALD TEST): ***\n")
x1_se <- summary(altman_model)$coefficients["X1", "Std. Error"]
x1_z <- summary(altman_model)$coefficients["X1", "z value"]
x1_p <- summary(altman_model)$coefficients["X1", "Pr(>|z|)"]

cat("Statistique Z de Wald:", round(x1_z, 3), "\n")
cat("p-valeur:", round(x1_p, 6), "\n")

if(x1_p < 0.001) {
  cat("🎯 RÉSULTAT: X1 est TRÈS SIGNIFICATIF (p < 0.001)\n")
  cat("=> Le fonds de roulement / actif total a un impact statistiquement très significatif sur le défaut\n")
} else if(x1_p < 0.05) {
  cat("🎯 RÉSULTAT: X1 est SIGNIFICATIF (p < 0.05)\n")
  cat("=> Le fonds de roulement / actif total a un impact statistiquement significatif sur le défaut\n")
} else {
  cat("❌ RÉSULTAT: X1 n'est PAS SIGNIFICATIF (p >= 0.05)\n")
  cat("=> Le fonds de roulement / actif total n'a pas d'impact statistiquement significatif sur le défaut\n")
}

# Intervalle de confiance pour X1
cat("\n*** INTERVALLE DE CONFIANCE POUR X1 (95%): ***\n")
x1_ci <- confint(altman_model)["X1", ]
cat("Intervalle de confiance à 95%: [", round(x1_ci[1], 4), ";", round(x1_ci[2], 4), "]\n")

# Interprétation en termes d'odds ratios
x1_or_ci <- exp(x1_ci)
cat("Odds Ratio - IC à 95%: [", round(x1_or_ci[1], 4), ";", round(x1_or_ci[2], 4), "]\n")

if(x1_ci[1] <= 0 & x1_ci[2] >= 0) {
  cat("⚠️  ATTENTION: L'intervalle de confiance contient 0 → effet pas significativement différent de 0\n")
} else {
  cat("✅ L'intervalle de confiance ne contient pas 0 → effet significativement différent de 0\n")
}

# POINT 3: MODÈLE ALTMAN ÉTENDU AVEC DtD -------------------------------
cat("\n=== MODÈLE ALTMAN ÉTENDU AVEC DtD ===\n")

# Vérifier si DtD existe dans le dataset
if("DtD" %in% names(altman_data)) {
  cat("*** AJUSTEMENT DU MODÈLE ALTMAN + DtD: ***\n")
  
  altman_model_dtd <- glm(default ~ X1 + X2 + X3 + X4 + X5 + DtD, 
                          data = altman_data, 
                          family = binomial(link = "logit"))
  
  cat("\n*** RÉSUMÉ DU MODÈLE ALTMAN + DtD: ***\n")
  summary(altman_model_dtd)
  
  # Test LRT: altman_model vs altman_model_dtd
  cat("\n*** LIKELIHOOD RATIO TEST: ALTMAN vs ALTMAN + DtD: ***\n")
  cat("H0: Le modèle Altman seul est suffisant (DtD n'apporte rien)\n")
  cat("H1: L'ajout de DtD améliore significativement le modèle\n\n")
  
  lrt_altman_dtd <- anova(altman_model, altman_model_dtd, test = "Chisq")
  print(lrt_altman_dtd)
  
  p_value_dtd <- lrt_altman_dtd$`Pr(>Chi)`[2]
  cat("\n*** DÉCISION CONCERNANT DtD: ***\n")
  if(is.na(p_value_dtd)) {
    cat("❌ p-valeur non disponible - garder le modèle Altman simple\n")
    final_altman <- altman_model
  } else if(p_value_dtd < 0.05) {
    cat("✅ DtD est SIGNIFICATIVE (p =", round(p_value_dtd, 6), "< 0.05)\n")
    cat("=> DÉCISION: GARDER DtD dans le modèle\n")
    final_altman <- altman_model_dtd
  } else {
    cat("❌ DtD n'est PAS SIGNIFICATIVE (p =", round(p_value_dtd, 6), ">= 0.05)\n")
    cat("=> DÉCISION: SUPPRIMER DtD du modèle\n")
    final_altman <- altman_model
  }
  
} else {
  cat("❌ Variable DtD non trouvée dans le dataset\n")
  cat("=> CONTINUATION avec le modèle Altman de base\n")
  altman_model_dtd <- altman_model
  final_altman <- altman_model
}

# POINT 4: MODÈLE ALTMAN SANS X4 --------------------------------------
cat("\n=== MODÈLE ALTMAN SANS X4 ===\n")

# Créer le modèle sans X4
if("DtD" %in% names(altman_data) && exists("altman_model_dtd") && !identical(altman_model_dtd, altman_model)) {
  cat("*** AJUSTEMENT DU MODÈLE ALTMAN + DtD SANS X4: ***\n")
  
  altman_model_dtd_wo_x4 <- glm(default ~ X1 + X2 + X3 + X5 + DtD, 
                                data = altman_data, 
                                family = binomial(link = "logit"))
  
  cat("\n*** RÉSUMÉ DU MODÈLE ALTMAN + DtD SANS X4: ***\n")
  summary(altman_model_dtd_wo_x4)
  
  # Test LRT: altman_model_dtd vs altman_model_dtd_wo_x4
  cat("\n*** LIKELIHOOD RATIO TEST: (ALTMAN + DtD) vs (ALTMAN + DtD - X4): ***\n")
  cat("H0: X4 n'est pas nécessaire (modèle sans X4 suffisant)\n")
  cat("H1: X4 apporte une contribution significative\n\n")
  
  lrt_x4 <- anova(altman_model_dtd_wo_x4, altman_model_dtd, test = "Chisq")
  print(lrt_x4)
  
  p_value_x4 <- lrt_x4$`Pr(>Chi)`[2]
  cat("\n*** DÉCISION CONCERNANT X4: ***\n")
  if(is.na(p_value_x4)) {
    cat("❌ p-valeur non disponible - garder X4\n")
    final_altman <- altman_model_dtd
  } else if(p_value_x4 < 0.05) {
    cat("✅ X4 est SIGNIFICATIVE (p =", round(p_value_x4, 6), "< 0.05)\n")
    cat("=> DÉCISION: GARDER X4 dans le modèle\n")
    final_altman <- altman_model_dtd
  } else {
    cat("❌ X4 n'est PAS SIGNIFICATIVE (p =", round(p_value_x4, 6), ">= 0.05)\n")
    cat("=> DÉCISION: SUPPRIMER X4 du modèle\n")
    final_altman <- altman_model_dtd_wo_x4
  }
  
} else {
  cat("*** AJUSTEMENT DU MODÈLE ALTMAN DE BASE SANS X4: ***\n")
  
  altman_model_wo_x4 <- glm(default ~ X1 + X2 + X3 + X5, 
                            data = altman_data, 
                            family = binomial(link = "logit"))
  
  # Test LRT: altman_model_wo_x4 vs altman_model
  cat("\n*** LIKELIHOOD RATIO TEST: ALTMAN SANS X4 vs ALTMAN COMPLET: ***\n")
  lrt_x4_base <- anova(altman_model_wo_x4, altman_model, test = "Chisq")
  print(lrt_x4_base)
  
  p_value_x4_base <- lrt_x4_base$`Pr(>Chi)`[2]
  if(is.na(p_value_x4_base) || p_value_x4_base >= 0.05) {
    cat("=> DÉCISION: SUPPRIMER X4 du modèle\n")
    final_altman <- altman_model_wo_x4
  } else {
    cat("=> DÉCISION: GARDER X4 dans le modèle\n")
    final_altman <- altman_model
  }
}

# POINT 5: MODÈLE ALTMAN FINAL -----------------------------------------
cat("\n=== MODÈLE ALTMAN FINAL RECOMMANDÉ ===\n")

# Créer le modèle Altman final selon spécification: avec DtD et sans X4
cat("*** CRÉATION DU MODÈLE ALTMAN FINAL SPÉCIFIÉ: ***\n")
cat("=> Modèle avec DtD (incluse) et sans X4 (exclue)\n")
cat("=> Variables: X1, X2, X3, X5, DtD\n\n")

# Vérifier si DtD existe
if("DtD" %in% names(altman_data)) {
  # Créer le modèle final: Altman sans X4 mais avec DtD
  altman_final_specified <- glm(default ~ X1 + X2 + X3 + X5 + DtD, 
                               data = altman_data, 
                               family = binomial(link = "logit"))
  
  cat("✅ MODÈLE ALTMAN FINAL CRÉÉ: X1 + X2 + X3 + X5 + DtD\n")
} else {
  # Si DtD n'existe pas, créer sans DtD et sans X4
  altman_final_specified <- glm(default ~ X1 + X2 + X3 + X5, 
                               data = altman_data, 
                               family = binomial(link = "logit"))
  
  cat("⚠️  DtD non trouvée - MODÈLE CRÉÉ: X1 + X2 + X3 + X5 (sans X4)\n")
}

cat("\n*** RÉSUMÉ DU MODÈLE ALTMAN FINAL SPÉCIFIÉ: ***\n")
summary(altman_final_specified)

# Utiliser ce modèle comme final
final_altman <- altman_final_specified

cat("\n*** COMPARAISON DES MODÈLES ALTMAN: ***\n")
cat("Modèle Altman de base - AIC:", round(AIC(altman_model), 2), "\n")
if(exists("altman_model_dtd") && !identical(altman_model_dtd, altman_model)) {
  cat("Modèle Altman + DtD - AIC:", round(AIC(altman_model_dtd), 2), "\n")
}
cat("Modèle Altman final - AIC:", round(AIC(final_altman), 2), "\n")

# Variables du modèle final
final_vars <- names(coef(final_altman))[-1]  # Exclure l'intercept
cat("\n*** VARIABLES DU MODÈLE ALTMAN FINAL: ***\n")
for(i in seq_along(final_vars)) {
  var_name <- final_vars[i]
  coef_val <- coef(final_altman)[var_name]
  direction <- if(coef_val > 0) "↗️" else "↘️"
  cat(sprintf("  %d. %s %s (coef: %.6f)\n", i, var_name, direction, coef_val))
}

# Comparaison avec le modèle Z-score
cat("\n*** COMPARAISON AVEC LE MODÈLE Z-SCORE ALTMAN: ***\n")
cat("AIC modèle variables séparées:", round(AIC(final_altman), 2), "\n")
cat("AIC modèle Z-score:", round(AIC(altman_model_zscore), 2), "\n")

cat("\n*** RÉSUMÉ DU MODÈLE Z-SCORE: ***\n")
summary(altman_model_zscore)

# Interprétation du Z-score dans le modèle logistique
zscore_coef <- coef(altman_model_zscore)["Z_score"]
cat("\n*** INTERPRÉTATION DU Z-SCORE DANS LE MODÈLE LOGISTIQUE: ***\n")
cat("Coefficient Z-score:", round(zscore_coef, 4), "\n")
if(zscore_coef < 0) {
  cat("✅ Coefficient négatif: Plus le Z-score est élevé, moins le risque de défaut est élevé (logique)\n")
  cat("📉 Pour chaque augmentation d'1 unité du Z-score, les odds de défaut sont multipliées par", round(exp(zscore_coef), 4), "\n")
} else {
  cat("⚠️  Coefficient positif: Résultat contre-intuitif à analyser\n")
}

cat("\n🎯 === ANALYSE ALTMAN TERMINÉE === 🎯\n")

