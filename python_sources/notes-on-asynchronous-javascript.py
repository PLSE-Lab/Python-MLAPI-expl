#!/usr/bin/env python
# coding: utf-8

# * The old style of asynchronous functions used callbacks, resulting in the well-known callback pyramid of doom.
# * Promises are the second generation of JavaScript language features addressing the asynchronous use cases (which, in JavaScript, is almost every use case).
# * The `async` and `await` keywords are the third generation of JavaScript language features addressing the asynchronous use case.
# * Modern code uses `async` and `await`. But it's important to understand how promises work.
# * Older JavaScript APIs, e.g. ones that predate the introduction of promises, still use the callback pattern. But these can (and should!) be wrapped in the promise or async API to make them modern and to have them "fit" with the rest of the application.
# 
# 
# * Functions which returns promises may be have their promises chained:
# 
# ```javascript
# doSomething()
# .then(function(result) {
#   return doSomethingElse(result);
# })
# .then(function(newResult) {
#   return doThirdThing(newResult);
# })
# .then(function(finalResult) {
#   console.log('Got the final result: ' + finalResult);
# })
# ```
# 
# * For error management, a `catch` block is also available. `catch` blocks can be mixed arbitrarily with `then` blocks to allow for logic to be performed in case of failure.
# * Errors flow down the method chain. An error in an earlier function is handled by the `catch` block nearest to it in downstream.
# 
# 
# * The `Promise.resolve()` class method intializes an already-resolved promise. The `Promise.reject()` class method initializes an already-failed promise. This has niche applications.
# * `Promise.all()` and `Promise.race()` can be used to create decision logic in promise resolution.
# * Something that I did not understand initially is when a promise is considered resolved. A `resolve` function is passed to the promise as its first input, *and it must be called in order for a promise to be considered resolved*. A promise whose method body contains logic with no `resolve` call stays queued forever (this took way too long to figure out...sigh). The `resolve` function can take a value; if you provide one the value passed to `then` is that value, otherwise the value passed to `then` is `null`.
# 
# ```javascript
# // a model asynchronous function
# const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
# wait().then(() => console.log(4)).then(() => console.log(5));
# 
# // prints 4, 5
# 
# // my own asynchronous function
# const lol = () => new Promise((resolve, reject) => { 
# console.log('hello world!');
#     resolve(5);
# });
# lol().then((v) => console.log(v)).then(() => console.log(5));
# 
# // prints 5 twice
# ```

# In[ ]:




