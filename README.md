# On the Robustness of Fully-Spiking Neural Networks in Open-World Scenarios using Forward-Only Learning Algorithms

In the last decade, Artificial Intelligence (AI) models have rapidly integrated into production pipelines propelled by their excellent modeling performance. However, the development of these models has not been matched by advancements in algorithms ensuring their safety, failing to guarantee robust behavior against Out-of-Distribution (OoD) inputs outside their learning domain. Furthermore, there is a growing concern with the sustainability of AI models and their required energy consumption in both training and inference phases. To mitigate these issues, this work explores the use of the Forward-Forward Algorithm (FFA), a biologically plausible alternative to the Backpropagation algorithm, adapted to the spiking domain to enhance the overall energy efficiency of the model. By capitalizing on the highly expressive topology emerging from the latent space of models trained with FFA, we develop a novel FF-SCP algorithm for OoD Detection. Our approach measures the likelihood of a sample belonging to the in-distribution (ID) data by using the distance from the latent representation of samples to class-representative manifolds. Additionally, to provide deeper insights into our OoD pipeline, we propose a gradient-free attribution technique that highlights the features of a sample pushing it away from the distribution of any class. Multiple experiments using our spiking FFA adaptation demonstrate that the achieved accuracy levels are comparable to those seen in analog networks trained via back-propagation. Furthermore, OoD detection experiments on multiple datasets (e.g., Omniglot and Not-MNIST) prove that FF-SCP outperforms avant-garde OoD detectors within the spiking domain in terms of several metrics used in this area, including AUROC, AUPR, and FPR-95. We also present a qualitative analysis of our explainability technique, exposing the precision by which the method detects OoD features, such as embedded artifacts or missing regions, in multiple instances of MNIST and KMNIST. Our results underscore the enhanced robustness that can be achieved by analyzing the latent spaces produced by already trained models.

## Launching the code

### Launching MNIST
```
python train.py --dataset "mnist" --neurons_per_layer "1400" --input_size "3072" --batch_size "512" --threshold "2" --dataset_resize "32"
```

### Launching OOD Experiments
```
python OOD_tests.py --dataset_group "small"
```