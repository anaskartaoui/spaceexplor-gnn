# SpaceExplor-GNN

A Geometric Deep Learning engine designed to predict high-value midfield space exploitation using coordinate-independent, field-symmetry invariant graph networks.

---

## The Problem: Traditional Models are "Pitch-Blind"

In modern football analytics, player tracking data is recorded as absolute grid coordinates $(x, y)$. When standard Machine Learning models (like standard Neural Networks or standard Graph Attention Networks) process these coordinates, they treat them as abstract numbers. 

This creates a massive flaw in spatial understanding:

* **No Inherent Geometry:** A standard model does not know that a football pitch is a symmetrical rectangle. 
* **Data Inefficiency:** If a midfielder, winger, and striker create a perfect passing triangle on the left flank, the model sees a specific set of numbers (e.g., $x=20, y=15$). If that exact same tactical triangle is mirrored on the right flank, the coordinates change completely (e.g., $x=80, y=55$). 
* **The Consequence:** A standard model views these identical tactical concepts as completely unrelated events. It requires thousands of extra data points just to learn that the game of football behaves the same way on both sides of the pitch.

---

## The Solution: Geometric Invariance

`SpaceExplor-GNN` solves this problem by enforcing **Field-Orientation Invariance** natively inside the neural network architecture. 

Instead of feeding absolute pitch coordinates into the model, we transform the tracking frames into a dynamic geometric mesh. The model processes **relative distances** and **local spatial angles** between players. 

### Why This Matters:
1. **Instant Symmetry Recognition:** If the model learns how a defensive block collapses on the left wing, it instantly understands how it collapses on the right wing, without needing new training data.
2. **True Space Identification:** It evaluates the physical "faces" of space (using Voronoi and Delaunay geometries) to predict the optimal coordinates a midfielder should occupy to break a defensive line.

---

## Tech Stack

* **Core Logic:** PyTorch Geometric (Graph Neural Networks)
* **Data Processing:** Polars (High-performance dataframe manipulation)
* **Ingestion:** Kloppy (Standardized sports tracking serialization)