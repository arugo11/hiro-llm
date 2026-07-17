TYPST_ENTRY ?= docs/main.typ
TYPST_OUTPUT ?= build/main.pdf

.PHONY: typst-build typst-check typst-watch clean

typst-build:
	@mkdir -p "$(dir $(TYPST_OUTPUT))"
	typst compile --root . "$(TYPST_ENTRY)" "$(TYPST_OUTPUT)"

typst-check: typst-build

typst-watch:
	@mkdir -p "$(dir $(TYPST_OUTPUT))"
	typst watch --root . "$(TYPST_ENTRY)" "$(TYPST_OUTPUT)"

clean:
	rm -rf build
