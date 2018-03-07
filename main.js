const Apify = require('apify'); // eslint-disable-line
const natural = require('natural'); // eslint-disable-line
// Local development & testing
const data = require('./data.json');

const { log } = console;

// Compound training for Bayes & Logistic Regression classifier.

// INPUT a serialize data structure to train,
// optional: object with the desired OUTPUT structure
// Restore previous training,
// Train with INPUT data,
// Return serialize classified results for both classifiers (OUTPUT),
// { logistic, bayes }
// Save previous training.

// To do: refactor to reduce
const keysAndValues = {};
function train(data, previousKey) {
  if (data && typeof data === 'object') {
    if (Array.isArray(data)) {
      data.forEach(item => train(item, previousKey));
    } else {
      const keys = Object.keys(data);
      keys.forEach(key => train(data[key], key));
    }
  } else if (data) {
    if (previousKey in keysAndValues) {
      keysAndValues[previousKey].push(data);
    } else {
      keysAndValues[previousKey] = [data];
    }
  }
}

Apify.main(async () => {
  const input = await Apify.getValue('INPUT');
  if (!data && !input.data) {
    throw new Error('data in INPUT is required.');
  }

  const classifiersType = ['LOGISTIC', 'BAYES'];
  const classifiers = classifiersType.map(
    classifier => Apify.getValue(classifier).then(JSON.parse)
  );
  const [previousLogistic, previousBayes] = await Promise.all(classifiers);

  let logistic;
  if (previousLogistic) {
    log('Restoring previous LOGISTIC...');
    logistic = natural.LogisticRegressionClassifier.restore(previousLogistic);
  } else {
    log('Creating new Logistic Regression CLASSIFIER.');
    logistic = new natural.LogisticRegressionClassifier();
  }

  let bayes;
  if (previousBayes) {
    log('Restoring previous BAYES...');
    bayes = natural.BayesClassifier.restore(previousBayes);
  } else {
    log('Creating new Bayes CLASSIFIER.');
    bayes = new natural.BayesClassifier();
  }

  // Training
  if (data.length) {
    log('Data found:', data.length);
    console.time('training-time');
    train(data);

    // Reduce the size of the data to train. O(n^2) complexity.
    Object.keys(keysAndValues).forEach((key) => {
      log('Training:', key);
      // Reduce the training data size
      const values = [...new Set(keysAndValues[key])].slice(0, 30);

      values.forEach((val) => {
        logistic.addDocument(val, key);
        bayes.addDocument(val, key);
      });
    });
    logistic.train();
    bayes.train();
    console.timeEnd('training-time');
  }

  const examples = ['03/5/2018', 'active', 'IL', 'MA'];
  examples.forEach((ex) => {
    const testLogistic = logistic.classify(ex);
    const testBayes = bayes.classify(ex);
    log(ex, testLogistic, testBayes);
  });
  log('Saving classifiers...');
  const savePreviousClassifiers = [logistic, bayes].map(
    (classifier, i) => Apify.setValue(classifiersType[i], JSON.stringify(classifier))
  );
  await Promise.all(savePreviousClassifiers);

  log('Done.');
});
