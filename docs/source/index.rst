.. LOOM documentation master file

====
LOOM
====

**Lowering Optimizer with Optimized Matching**

LOOM is a WebAssembly optimizer using ISLE (Instruction Selection/Lowering Expressions)
with formal verification capabilities. It aims to provide a proof-ready alternative
to wasm-opt from Binaryen.

.. image:: https://img.shields.io/badge/status-proof%20of%20concept-yellow
   :alt: Status: Proof of Concept

.. image:: https://img.shields.io/badge/verification-SMT%20based-blue
   :alt: Verification: SMT Based

Overview
========

LOOM combines:

- **ISLE DSL** from Cranelift for declarative term-rewriting optimization rules
- **Crocus** SMT-based verification for proving optimization soundness
- **Binaryen compatibility** for leveraging existing test infrastructure
- **Component Model** support for modern WebAssembly applications

Key Features
============

Formal Verification
-------------------

Every optimization rule in LOOM can be formally verified using SMT solvers:

- Automatic counterexample generation for unsound rules
- Bitvector modeling for all WebAssembly value types
- Memory operation verification
- Control flow transformation proofs

Declarative Optimizations
--------------------------

Write optimizations as simple pattern-matching rules:

.. code-block:: isle

   ;; Constant folding for integer addition
   (rule (lower (iadd (iconst x) (iconst y)))
         (iconst (iadd_imm x y)))

Test Integration
----------------

- Compatible with Binaryen's test harness
- Fuzzing support for finding edge cases
- Semantic equivalence testing

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   requirements/index
   architecture/index
   development/index

Project Status
==============

LOOM is currently in the **proof of concept** phase. See the :doc:`requirements/index`
for detailed status of each component.

Quick Links
===========

* :doc:`requirements/index` - All project requirements
* :doc:`architecture/index` - System architecture
* :doc:`development/index` - Development guide

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`
