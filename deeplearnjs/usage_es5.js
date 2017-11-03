var dl = deeplearn;
var math = new dl.NDArrayMathGPU();
var a = dl.Array1D.new([1, 2, 3]);
var b = dl.Scalar.new(2);

var result = math.add(a, b);

// for(var propertyName in result) {console.log(typeof propertyName, propertyName)}

// // Option 1: With a Promise.
// result.getValuesAsync().then(data => console.log(data)); // Float32Array([3, 4, 5])

// // Option 2: Synchronous download of data. This is simpler, but blocks the UI.
// console.log(result.getValues());


// Option 1: With a Promise.
result.data().then(data => console.log(data));

// Option 2: Synchronous download of data. This is simpler, but blocks the UI.
console.log(result.dataSync());