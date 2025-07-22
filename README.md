# BetaBoards: Predicting Indoor Climbing Difficulty with Graph-Based Modeling

## Project Summary

**BetaBoards** introduces a novel, scalable method for estimating indoor climbing route difficulty using graph-based structural modeling. Instead of relying on computer vision or fixed-wall systems, this approach converts climbing routes into graphs and extracts interpretable features—such as vertical gain, hold size, and move distances—to train traditional machine learning models.

This project was developed as part of the RAID Data Challenge by Byron Bhuiyan, Alan Xu-Zhang, and Shawn Hwang at the University of California, Riverside.

## Key Features

- Graph-based route modeling using spatially feasible hold transitions
- Interpretable structural and geometric feature extraction
- Weak supervision via color-to-grade mapping for label generation
- Lightweight pipeline deployable with simple annotated hold data
- Machine learning model trained on real gym wall data

## Methodology

### Route Representation

Each climbing route is modeled as a graph `G = (V, E)`:
- Nodes (`V`): Holds with (x, y) positions, bounding box size, and color
- Edges (`E`): Connect holds of the same color within a normalized Euclidean distance threshold (0.15)

Each route color is processed as a separate subgraph.

### Feature Extraction

For each subgraph (route), the following features are computed:

**Structural Features**
- Number of nodes
- Number of edges
- Graph density
- Average degree
- Clustering coefficient
- Graph diameter

**Geometric Features**
- Average move distance
- Maximum move distance
- Vertical gain
- Horizontal spread

**Hold-Specific Features**
- Average hold size
- Proportion of small holds
- Number of long moves (> 0.2 normalized distance)

### Difficulty Labeling

Due to the lack of explicit difficulty ratings, route difficulty is weakly supervised using gym-specific color-to-grade mappings (e.g., green = V1, red = V5).

## Results

- **Model**: Random Forest classifier
- **Evaluation**: 5-fold stratified cross-validation
- **Accuracy**: 
  - Cross-validation: ~20.7% (due to limited data and class imbalance)
  - Full dataset: **82.7%**
- **Top Predictive Features**: Number of holds, average hold size, vertical gain, and maximum move distance
- **Key Insight**: Structural graph features align well with real-world difficulty perceptions

## Limitations

- Weak supervision may introduce labeling noise due to color inconsistencies
- Wall angle, hold orientation, and texture are not currently modeled
- Dataset is limited to a single gym wall and requires broader validation

## Future Work

- Collect a larger, more diverse dataset across multiple gym walls
- Incorporate wall angle and other physical metadata
- Explore Graph Neural Networks (GNNs) for learning from climbing sequences
- Integrate climber feedback or crowd-sourced grading for label refinement

## Citation
 Byron Bhuiyan, Alan Xu-Zhang, and Shawn Hwang. 2018. Raid Data Challenge. Proc. ACM Meas. Anal. Comput. Syst. 37, 4,
 Article 111 (August 2018), 7 pages. https://doi.org/XXXXXXX.XXXXXX
 Proc. ACM Meas. Anal. Comput. Syst., Vol. 37, No. 4, Article 111. Publication date: August 2018
## License

This project is licensed under the MIT License.
