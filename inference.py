import tensorflow as tf
from model import create_model

def plot_prediction(model, X_test, y_test, num_predictions):
    fig=plt.figure(figsize=(20, (num_predictions/6 * 25)))
    for i in range(0, num_predictions):
        image_num = random.randint(0, X_test.shape[0])
        img = X_test[image_num]
        img = np.reshape(img, (256, 256))
        
        real_mask = np.argmax(y_test[image_num], axis=2)

        prediction = model.predict(np.array([X_test[image_num]]))
        predicted_mask = np.argmax(prediction[0], axis=2)

        ax = fig.add_subplot(num_predictions, 3, 1 + (i*3))
        ax.axis('off')
        plt.imshow(img)
        ax = fig.add_subplot(num_predictions, 3, 2 + (i*3))
        ax.axis('off')
        plt.imshow(real_mask)
        ax = fig.add_subplot(num_predictions, 3, 3 + (i*3))
        ax.axis('off')
        plt.imshow(predicted_mask)
    plt.show()

model = tf.keras.models.load_model('trained_model.h5')


