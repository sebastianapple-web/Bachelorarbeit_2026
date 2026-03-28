# Bachelorarbeit_2026

## Projektbeschreibung
Dieses Repository enthält den Programmcode, die Konfigurationsdateien, die Auswertungsroutinen und ausgewählte Ergebnisdateien zur Bachelorarbeit
„Über den Einfluss der Sampling-Temperatur auf Halluzinationen in Retrieval-Augmented-Generation-gestützten Large Language Models“.

## Inhalt des Repositorys
- `src/` – Skripte zur Datenvorbereitung, Evaluation, Aggregation und Visualisierung
- `docs/` – Dokumentation des Experiments und der Konfiguration
- `regression_outputs/` – Skripte und Abhängigkeiten für die Regressionsauswertung
- `llama_cpp_run_20260312_184456_500/` – ausgewählte Ergebnisdateien und Plots eines Experimentlaufs
- `corpus/` – ausgewählte Korpusdateien ohne große bzw. nicht veröffentlichte Dateien
- `questions_500.jsonl` – verwendeter Fragenkorpus

## Voraussetzungen
- Python 3.x
- weitere Abhängigkeiten gemäß `requirements.txt`

## Installation
1. Repository klonen
2. Abhängigkeiten installieren
3. Skripte im Ordner `src/` bzw. `regression_outputs/` ausführen

## Nutzung
Das Repository dient der Nachvollziehbarkeit des experimentellen Setups, der Durchführung der Auswertung und der Reproduktion zentraler Ergebnisdarstellungen.

## Hinweise
Große Modelldateien sowie einzelne große Korpusdateien sind aus technischen Gründen nicht im Repository enthalten.

## Reproduzierbarkeit
Der für die Abgabe relevante Projektstand wird über den Git-Commit-Hash eindeutig versioniert. Das Repository dient der technischen Nachvollziehbarkeit des experimentellen Setups, der Auswertungsschritte und der Ergebnisgenerierung. Nicht im Repository enthaltene große Dateien, insbesondere Modelldateien und umfangreiche Korpusdateien, müssen für eine vollständige Reproduktion separat lokal bereitgestellt werden.

Die im Experiment verwendete Modelldatei ist ebenfalls nicht Bestandteil des Repositorys. Für die Reproduktion muss das verwendete Modell (https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf) lokal separat bereitgestellt und im vorgesehenen Verzeichnis abgelegt werden. Dokumentiert sind der verwendete Modelltyp, das Dateiformat und die Konfiguration des Experiments, nicht jedoch die Binärdatei selbst.

## Autor
Sebastian Weindl
