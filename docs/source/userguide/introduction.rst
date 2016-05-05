.. _ug-introduction:

Introduction
============
This user guide is a general introduction to MenpoFit, aiming to provide a
bird's eye of MenpoFit's design. After reading this guide you should be able to
go explore MenpoFit's extensive Notebooks and not be too surprised by what you
see.

What makes MenpoFit better?
---------------------------
The vast majority of **existing deformable modeling software** suffers from one
or more of the following important issues:

- It is released in binary closed-source format
- It does not come with training code; only pre-trained models
- It is not well-structured which makes it very difficult to tweak and alter
- It only focuses on a single method/model

**MenpoFit** overcomes the above issues by providing open-source *training*
and *fitting* code for multiple state-of-the-art deformable models under a
unified protocol. We **strongly** believe that this is the only way towards
reproducable and high-quality research.

Core Interfaces
---------------
MenpoFit is an object oriented framework for building and fitting deformable
models. It makes some basic assumptions that are common for all the
implemented methods. For example, all deformable models are trained in
*multiple scales* and the fitting procedure is, in most cases, *iterative*.
MenpoFit's key interfaces are:

- :map:`MultiScaleNonParametricFitter` - multi-scale fitting class
- :map:`MultiScaleParametricFitter` - multi-scale fitting class that uses a parametric shape model
- :map:`MultiScaleNonParametricIterativeResult` - multi-scale result of an iterative fitting
- :map:`MultiScaleParametricIterativeResult` - multi-scale result of an iterative fitting using a parametric shape model

Deformable Models
-----------------
- :map:`AAM`, :map:`LucasKanadeAAMFitter`, :map:`SupervisedDescentAAMFitter` - Active Appearance Model builder and fitters
- :map:`ATM`, :map:`LucasKanadeATMFitter` - Active Template Model builder and fitter
- :map:`GenerativeAPS`, :map:`GaussNewtonAPSFitter` - Active Pictorial Structures builder and fitter
- :map:`CLM`, :map:`GradientDescentCLMFitter` - Constrained Local Model builder and fitter
- :map:`LucasKanadeFitter` - Lucas-Kanade Image Alignment
- :map:`SupervisedDescentFitter` - Supervised Descent Method builder and fitter
- :map:`DlibERT` - Ensemble of Regression Trees builder and fitter
