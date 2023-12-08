# Landiv

## Previews

![France-CH border](./test_france.png)

## Individual layers

![France-CH border](./test_france_layers.png)

### Individual layers with Gaussian filter

![France-CH border](./test_france_layers_filtered_1.0.png)
_sigma = 1_
![France-CH border](./test_france_layers_filtered_10.0.png)
_sigma = 10_
![France-CH border](./test_france_layers_filtered_40.0.png)
_sigma = 40_

## Entropy after diffusion

![France-CH border](./test_france.png)
![France-CH border](./test_france_layers_entropy_1.0.png)
_sigma = 1_

---

![France-CH border](./test_france_layers_entropy_10.0.png)
_sigma = 10_

---

![France-CH border](./test_france_layers_entropy_40.0.png)
_sigma = 40_

---

<br>

<p align="center">
<img 
   alt="Test area FR-CH border"
   src="./test_france.png" 
   height="900"
/>
<img 
  alt="Test area FR-CH border - entropy"
  src="./test_france_layers_entropy_40.0.png" 
  height="900"
/>
</p>

<br>

---
---

# Bigger map

<br>

<p align="center">
<img 
   alt="all france"
   src="./all_france.png" 
   height="900"
/>
<img 
  alt="All france - entropy sigma 200"
  src="./all_france_layers_entropy_200.0.png" 
  height="900"
/>
</p>

<br>


In principle this approach can be adapted also for landscape blocks consisting of a block of pixels and thus an initial distributions with resulting entropy.
Therefore, there are two way to include scale effects:

- the standard deviation of the diffusion kernel
- the landscape block size
