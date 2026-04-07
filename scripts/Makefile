PYTHON = python3
WAPIC = ../build/wapic
DATA = ../data
PATTERN = $(DATA)/pattern.txt
MODEL = $(DATA)/model_v2.crf

# Training params
ALGO = l-bfgs
MAXITER = 100
L1 = 0.5
L2 = 0.0001

all: review

# Step 1: prepare BMES training/test data from raw JSON
prepare: $(DATA)/train.txt $(DATA)/test.txt

$(DATA)/train.txt $(DATA)/test.txt: $(DATA)/1998-01.txt
	$(PYTHON) prepare_data.py

# Step 2: train CRF model
fit: $(MODEL)

$(MODEL): $(DATA)/train.txt $(PATTERN)
	$(WAPIC) train \
		-p $(PATTERN) \
		-a $(ALGO) \
		-i $(MAXITER) \
		-1 $(L1) \
		-2 $(L2) \
		$(DATA)/train.txt $(MODEL)

# Step 3: label test data
test: $(DATA)/test_result.txt

$(DATA)/test_result.txt: $(MODEL) $(DATA)/test_nolabel.txt
	$(WAPIC) label \
		-m $(MODEL) \
		$(DATA)/test_nolabel.txt $(DATA)/test_result.txt

# Step 4: evaluate
review: $(DATA)/test_result.txt $(DATA)/test.txt
	$(PYTHON) evaluate.py

clean:
	rm -f $(DATA)/train.txt $(DATA)/test.txt $(DATA)/test_nolabel.txt
	rm -f $(DATA)/test_result.txt

.PHONY: all prepare fit test review clean
