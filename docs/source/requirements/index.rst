============
Requirements
============

This section contains all requirements for the LOOM project, organized by category.

Requirements Overview
=====================

LOOM requirements are organized into the following categories:

.. toctree::
   :maxdepth: 2

   core
   verification
   testing
   optimizations_dce
   optimizations_constants
   optimizations_inlining
   optimizations_locals
   optimizations_memory
   optimizations_structure
   optimizations_instructions
   optimizations_globals
   optimizations_types
   optimizations_loops
   optimizations_dataflow
   optimizations_gc
   optimizations_lowering
   optimizations_js
   optimizations_special
   optimizations_debug
   optimizations_reordering
   optimizations_minification
   optimizations_eh
   optimizations_misc
   component_model

Requirements Status
===================

.. needpie::
   :labels: Planned, Active, Implemented, Verified, Complete
   :legend:

All Requirements Table
======================

.. needtable::
   :columns: id, title, status, priority, category
   :style: datatables

Requirements by Priority
========================

Critical Priority
-----------------

.. needtable::
   :filter: priority == "Critical"
   :columns: id, title, status, category
   :style: table

High Priority
-------------

.. needtable::
   :filter: priority == "High"
   :columns: id, title, status, category
   :style: table

Medium Priority
---------------

.. needtable::
   :filter: priority == "Medium"
   :columns: id, title, status, category
   :style: table

Requirements by Status
======================

Planned
-------

.. needtable::
   :filter: status == "planned"
   :columns: id, title, priority, category
   :style: table

Active
------

.. needtable::
   :filter: status == "active"
   :columns: id, title, priority, category
   :style: table

Implemented
-----------

.. needtable::
   :filter: status == "implemented"
   :columns: id, title, priority, category
   :style: table

Verified
--------

.. needtable::
   :filter: status == "verified"
   :columns: id, title, priority, category
   :style: table
