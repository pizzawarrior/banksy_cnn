## Banksy Image Authentication Using Convolutional Neural Network
Can we authenticate Banksy's street art using machine learning? Let's find out.

#### Dataset:
- 29 Authentic Images
- 27 Not Banksy Images, including forgeries and other artists with similar styles
- 44 total images used in training, randomly selected
- 12 total images used for testing, randomly selected. 6 images are attributable to Banksy, and 6 are not.

#### Confusion Matrix for the best 3-layer model:
<img width="500" alt="conf matrix" src="https://github.com/user-attachments/assets/82ecbd72-2a02-40c4-8f03-35a01a387876" />

#### Best 3-layer model performance:
- Accuracy: 75%
- F1: 73%
- Precision: 80%
- Recall: 67%
- AUC: 83%

These results indicate that the model was able to detect real Banksy images 2/3 times, and not-Banksy images 5/6 times.  
1 false positive out of 12 test images shows that this model has value as a tool in validating authentic artworks.

---
To run this project, from the root:
- activate .venv
- run `python -m main`
