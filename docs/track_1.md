# 2nd Workshop on Open-Vocabulary 3D Scene Understanding 

<h2><strong>Challenge Track 1</strong>: Open-vocabulary 3D object instance search</h2>

<!-- ![Alt text](assets/scenefun3d_2.png "a title") -->
<!-- <video controls autoplay loop poster="assets/teaser2_poster.png">
  <source src="assets/teaser2.mp4" type="video/mp4">
</video> -->

<!-- ![Track 1 teaser](assets/track1_teaser.png) -->
<p align="center">
<img src="/assets/track1_teaser.png" alt="Track 1 teaser" width="620"/>
</p>


## Overview 

<div style="text-align: justify">
The ability to perceive, understand and interact with arbitrary 3D environments is a long-standing research goal with applications in AR/VR, robotics, health and industry. Many 3D scene understanding methods are largely limited to recognizing a closed-set of pre-defined object classes. In the first track of our workshop challenge, we focus on open-vocabulary 3D object instance search. Given a 3D scene and an open-vocabulary, text-based query, the goal is to localize and densely segment all object instances that fit best with the specified query. If there are multiple objects that fit the given prompt, each of these objects should be segmented, and labeled as separate instances. The list of queries can refer to long-tail objects, or can include descriptions of object properties such as semantics, material type, and situational context.
</div>

## Tentative dates

- Submission Portal: EvalAI
- Data Instructions & Helper Scripts: April 15, 2024
- Dev Phase Start: April 15, 2024
- Submission Portal Start: April 15, 2024
- Test Phase Start: May 1, 2024
- Test Phase End: June 8, 2024 (23:59 Pacific Time)


## Task description

In the second track of our workshop challenge, we propose the following challenge:

>**TASK:** Given an open-vocabulary, text-based query, the aim is to localize and segment the object instances that fit best with the given prompt, which might describe object properties such as semantics, material type, affordances and situational context. 

>**INPUT:**  An RGB-D sequence and the 3D reconstruction of a given scene, camera parameters, and a text-based input query.

>**OUTPUT:** Instance segmentation of the point cloud that corresponds to the vertices of the provided 3D mesh reconstruction, segmenting the objects that fit best with the given prompt.

<!-- * `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit. -->

<!-- ## Data download

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.


## Submission instructions


## Evaluation guidelines -->