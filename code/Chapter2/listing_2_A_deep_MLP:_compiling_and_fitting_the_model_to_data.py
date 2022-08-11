model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32,
[CA]validation_split=0.1, shuffle=True,verbose=2)
