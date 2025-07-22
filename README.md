# Climbing
Raid Data Challenge
BYRON BHUIYAN, University of California, Riverside, USA
ALAN XU-ZHANG, University of California, Riverside, USA
SHAWN HWANG, University of California, Riverside, USA
Indoor climbing route grading is traditionally a subjective process, often relying on the intuition of route setters or informal
climber feedback. This subjectivity can lead to inconsistencies that affect climber experience and hinder skill progression.
In this work, we propose a novel, interpretable method for predicting indoor climbing route difficulty using graph-based
structural modeling. Each route is represented as a graph, where nodes correspond to holds and edges represent feasible
transitions based on spatial proximity. We extract interpretable features from these graphs — such as average move distance,
vertical gain, and hold size — and train machine learning models to predict route difficulty levels.
To address the lack of ground-truth labels, we apply a weak supervision strategy that maps hold color to approximate
V-grades using gym-specific conventions. Our experiments show that structural features alone are sufficient to predict
difficulty with high accuracy, and that the most predictive features align closely with physical challenge as experienced by
climbers. This work offers a lightweight, scalable alternative to image-based grading systems and provides a foundation for
future research in sports analytics, automated route assessment, and human-centered route design.
Additional Key Words and Phrases: Machine Learning, Data Science, LLM, AI Generated Text
ACM Reference Format:
Byron Bhuiyan, Alan Xu-Zhang, and Shawn Hwang. 2018. Raid Data Challenge. Proc. ACM Meas. Anal. Comput. Syst. 37, 4,
Article 111 (August 2018), 7 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Indoor climbing is rapidly growing in popularity, with thousands of gyms worldwide setting routes tailored to
climbers of varying skill levels. A key aspect of the climbing experience is route grading — assigning a difficulty
level to each route to help climbers select appropriate challenges and track progression. However, current grading
practices are highly subjective and inconsistent, often relying on a route setter’s judgment or climber feedback,
which can vary widely across gyms and regions.
This paper proposes a graph-based framework for predicting the difficulty of indoor climbing routes using
interpretable structural features. Unlike computer vision models that rely on large image datasets or fixed setups
(e.g., MoonBoard), our approach builds lightweight graphs from route hold data and extracts features that align
with human intuition about difficulty (e.g., move distances, vertical gain). Using a weakly-supervised dataset
collected from a real gym wall, we train machine learning models to classify routes by their estimated grade and
analyze which structural attributes most contribute to difficulty.
Authors’ addresses: Byron Bhuiyan, University of California, Riverside, Riverside, USA, rbhui003@ucr.edu; Alan Xu-Zhang, University of
California, Riverside, Riverside, USA, axuzh001@ucr.edu; Shawn Hwang, University of California, Riverside, Riverside, USA, shwan068@ucr.
edu.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that
copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page.
Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy
otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from
permissions@acm.org.
© 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 2476-1249/2018/8-ART111
https://doi.org/XXXXXXX.XXXXXXX
Proc. ACM Meas. Anal. Comput. Syst., Vol. 37, No. 4, Article 111. Publication date: August 2018.
111:2 • Trovato et al.
Our method offers a scalable, interpretable, and reproducible alternative to traditional grading, with potential
benefits for setters, climbers, and gym operators.
2 Related Work
Prior research into automated climbing difficulty assessment has focused largely on two areas: computer vision
and fixed-board learning systems. Image-based methods often involve detecting holds and analyzing their layout
through segmentation or pose estimation [1]. However, these methods require large labeled datasets and are
sensitive to visual noise, lighting, and wall complexity. Moreover, they often lack interpretability.
MoonBoard-based research has leveraged standardized boards with known hold layouts and sequences, enabling
sequence-based modeling with Recurrent Neural Networks or Graph Neural Networks (GNNs) [2]. While effective,
these models are limited in scope and cannot generalize to arbitrary wall configurations or setter styles.
Our work diverges by:
• Modeling general gym routes using graph structures built from simple positional data.
• Using structural and geometric features for interpretable classification.
• Avoiding reliance on images or fixed-wall formats, making the method more broadly applicable.
3 Methodology
3.1 Route Representation
Each climbing route is modeled as a graph𝐺 = (𝑉 , 𝐸) where V is the set of holds in the route, each node annotated
with:
• Normalized (x, y) position
• Hold size (from bounding box area)
• Hold color (used as a proxy for difficulty)
E represents feasible transitions between holds, created by adding edges between any two holds of the same
color that are within a specified Euclidean distance threshold (0.15 in normalized units). This simulates physical
reachability between moves. Since each image contains multiple route colors, we construct separate subgraphs per
color. This ensures that each route is evaluated independently, prevents cross-route interference, and maintains
alignment with how climbers follow single-color problems in practice.
