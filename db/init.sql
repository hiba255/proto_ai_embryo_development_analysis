CREATE TABLE IF NOT EXISTS analyses (
  id SERIAL PRIMARY KEY,
  nom_video VARCHAR(255) NOT NULL,
  date_analyse TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  resultat JSONB NOT NULL,
  confiance_moyenne REAL,
  version_modele VARCHAR(50)
);