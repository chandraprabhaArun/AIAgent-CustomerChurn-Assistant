.PHONY: setup train evaluate run clean

# Setup virtual environment and install dependencies
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Setup complete. Run: source venv/bin/activate"

# Train the churn model
train:
	python scripts/train_model.py

# Run evaluation pipeline (requires OPENAI_API_KEY)
evaluate:
	python scripts/run_evaluation.py

# Launch Streamlit app locally
run:
	streamlit run part2/app/main.py

# Run notebook end-to-end
notebook:
	jupyter nbconvert --execute part1/churn_model.ipynb --to notebook --inplace

# Clean generated artifacts
clean:
	rm -f models/*.joblib
	rm -f data/cleaned_data.csv
	rm -f part2/evaluation/results/scorecard.json
	@echo "🧹 Cleaned all generated artifacts"

# Full pipeline: train → evaluate → run
all: train evaluate run
