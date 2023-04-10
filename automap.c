// TODO: Rewrite performance tests using pyperf.
// TODO: Group similar functionality.
// TODO: Check refcounts when calling into hash and comparison functions.
// TODO: Check allocation and cleanup.
// TODO: Subinterpreter support.
// TODO: Docstrings and stubs.
// TODO: GC support.


/*******************************************************************************

Our use cases differ significantly from Python's general-purpose dict type, even
when setting aside the whole immutable/grow-only and contiguous-integer-values
stuff.

What we don't care about:

  - Memory usage. Python's dicts are used literally everywhere, so a tiny
    reduction in the footprint of the average dict results in a significant gain
    for *all* Python programs. We are happy to instead trade a few extra bytes
    of RAM for a more cache-friendly hash table design. Since we don't store
    values, we are still close to the same size on average!

  - Worst-case performance. Again, Python's dicts are used for literally
    everything, so they need to be able to gracefully handle lots of hash
    collisions, whether resulting from bad hash algorithms, heterogeneous keys
    with badly-combining hash algorithms, or maliciously-formed input. We can
    safely assume that our use cases don't need to worry about these issues, and
    instead choose lookup and collision resolution strategies that utilize cache
    lines more effectively. This extends to the case of lookups for nonexistent
    keys as well; we can assume that if our users are looking for something,
    they know that it's probably there.

What we do care about:

  - Creation and update time. This is *by far* the most expensive operation you
    do on a mapping. More on this below.

  - The speed of lookups that result in hits. This is what the mapping is used
    for, so it *must* be good. More on this below.

  - Iteration order and speed. You really can't beat a Python list or tuple
    here, so we can just store the keys in one of them to avoid reinventing the
    wheel. We use a list since it allows us to grow more efficiently.

So what we need is a hash table that's easy to insert into and easy to scan.

Here's how it works. A vanilla Python dict of the form:

{a: 0, b: 1, c: 2}

...basically looks like this (assume the hashes are 3, 6, and 9):

Indices: [-, 2, -, 0, -, -, 1,  -]

Hashes:  [3, 6, 9, -, -]
Keys:    [a, b, c, -, -]
Values:  [0, 1, 2  -, -]

It's pretty standard; keys, values, and cached hashes are stored in sequential
order, and their offsets are placed in the Indices table at position
HASH % TABLE_SIZE. Though it's not used here, collisions are resolved by jumping
around the table according to the following recurrence:

NEXT_INDEX = (5 * CURRENT_INDEX + 1 + (HASH >>= 5)) % TABLE_SIZE

This is good in the face of bad hash algorithms, but is sorta expensive. It's
also unable to utilize cache lines at all, since it's basically random (it's
literally based on random number generation)!

To contrast, the same table looks something like this for us:

Indices: [-, -, -, 0, -, -, 1, -, -, 2, -, -, -, -, -, -, -, -, -]
Hashes:  [-, -, -, 3, -, -, 6, -, -, 9, -, -, -, -, -, -, -, -, -]

Keys:    [a,  b,  c]

Right away you can see that we don't need to store the values, because they
match the indices (by design).

Notice that even though we allocated enough space in our table for 19 entries,
we still insert them into initial position HASH % 4.  This leaves the whole
15-element tail chunk of the table free for colliding keys. So, what's a good
collision-resolution strategy?

NEXT_INDEX = CURRENT_INDEX + 1

It's just a sequential scan! That means *every* collision-resolution lookup is
hot in L1 cache (and can even be predicted and speculatively executed). The
indices and hashes are actually interleaved for better cache locality as well.

We repeat this scan 15 times. We don't even have to worry about wrapping around
the edge of the table during this part, since we've left enough free space
(equal to the number of scans) to safely run over the end. It's wasteful for a
small example like this, but for more realistic sizes it's just about perfect.

We then jump to another spot in the table using a version of the recurrence
above:

NEXT_INDEX = (5 * (CURRENT_INDEX - 15) + 1 + (HASH >>= 1)) % TABLE_SIZE

...and repeat the whole thing over again. This collision resolution strategy is
similar to what Python's sets do, so we still handle some nasty collisions and
missing keys well.

There are a couple of other tricks that we use (like globally caching integer
objects from value lookups), but the hardware-friendly hash table design is what
really gives us our awesome performance.

*******************************************************************************/
# include <math.h>
# define PY_SSIZE_T_CLEAN
# include "Python.h"

# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"
# include "numpy/arrayscalars.h"
# include "numpy/halffloat.h"

# define DEBUG_MSG_OBJ(msg, obj)      \
    fprintf(stderr, "--- %s: %i: %s: ", __FILE__, __LINE__, __FUNCTION__); \
    fprintf(stderr, #msg " ");        \
    PyObject_Print(obj, stderr, 0);   \
    fprintf(stderr, "\n");            \
    fflush(stderr);                   \

//------------------------------------------------------------------------------
// Common

static PyTypeObject AMType;
static PyTypeObject FAMIType;
static PyTypeObject FAMVType;
static PyTypeObject FAMType;
static PyObject *NonUniqueError;


// The main storage "table" is an array of TableElement
typedef struct TableElement{
    Py_ssize_t keys_pos;
    Py_hash_t hash;
} TableElement;


// Table configuration; experimentation shows that these values work well:
# define LOAD 0.9
# define SCAN 16


typedef enum KeysArrayType{
    KAT_LIST = 0, // must be falsy

    KAT_INT8, // order matters as ranges of size are used in selection
    KAT_INT16,
    KAT_INT32,
    KAT_INT64,

    KAT_UINT8,
    KAT_UINT16,
    KAT_UINT32,
    KAT_UINT64,

    KAT_FLOAT16,
    KAT_FLOAT32,
    KAT_FLOAT64,

    KAT_UNICODE,
    KAT_STRING,
} KeysArrayType;


KeysArrayType
at_to_kat(int array_t) {
    switch (array_t) {
        case NPY_INT64:
            return KAT_INT64;
        case NPY_INT32:
            return KAT_INT32;
        case NPY_INT16:
            return KAT_INT16;
        case NPY_INT8:
            return KAT_INT8;

        case NPY_UINT64:
            return KAT_UINT64;
        case NPY_UINT32:
            return KAT_UINT32;
        case NPY_UINT16:
            return KAT_UINT16;
        case NPY_UINT8:
            return KAT_UINT8;

        case NPY_FLOAT64:
            return KAT_FLOAT64;
        case NPY_FLOAT32:
            return KAT_FLOAT32;
        case NPY_FLOAT16:
            return KAT_FLOAT16;

        case NPY_UNICODE:
            return KAT_UNICODE;
        case NPY_STRING:
            return KAT_STRING;
        default:
            return KAT_LIST;
    }
}


typedef struct FAMObject{
    PyObject_VAR_HEAD
    Py_ssize_t table_size;
    TableElement *table;    // an array of TableElement structs
    PyObject *keys;
    KeysArrayType keys_array_type;
    Py_ssize_t keys_size;
    Py_UCS4* key_buffer;
} FAMObject;


typedef enum ViewKind{
    ITEMS,
    KEYS,
    VALUES,
} ViewKind;


// NOTE: would like to use strchr(str, '\0') instead of this routine, but some buffers might not have a null terminator and stread by full to the the dt_size.
static inline Py_UCS4*
ucs4_get_end_p(Py_UCS4* p, Py_ssize_t dt_size) {
    Py_UCS4* p_end = p + dt_size;
    while (p < p_end && *p != '\0') {
        p++;
    }
    return p;
}


static inline char*
char_get_end_p(char* p, Py_ssize_t dt_size) {
    char* p_end = p + dt_size;
    while (p < p_end && *p != '\0') {
        p++;
    }
    return p;
}


static inline Py_hash_t
uint_to_hash(npy_uint64 v) {
    Py_hash_t hash = (Py_hash_t)(v >> 1); // half unsigned fits in signed
    if (hash == -1) {
        return -2;
    }
    return hash;
}

static inline Py_hash_t
int_to_hash(npy_int64 v) {
    Py_hash_t hash = (Py_hash_t)v;
    if (hash == -1) {
        return -2;
    }
    return hash;
}


// This is a adapted from https://github.com/python/cpython/blob/ba65a065cf07a7a9f53be61057a090f7311a5ad7/Python/pyhash.c#L92
#define HASH_MODULUS (((size_t)1 << 61) - 1)
#define HASH_BITS 61
static inline Py_hash_t
double_to_hash(double v)
{
    int e, sign;
    double m;
    Py_uhash_t x, y;

    if (isinf(v)) {
        return v > 0 ? 314159 : -314159;
    }
    if (isnan(v)) {
        return 0;
    }
    m = frexp(v, &e);
    sign = 1;
    if (m < 0) {
        sign = -1;
        m = -m;
    }
    x = 0;
    while (m) {
        x = ((x << 28) & HASH_MODULUS) | x >> (HASH_BITS - 28);
        m *= 268435456.0;  /* 2**28 */
        e -= 28;
        y = (Py_uhash_t)m;  /* pull out integer part */
        m -= y;
        x += y;
        if (x >= HASH_MODULUS)
            x -= HASH_MODULUS;
    }
    e = e >= 0 ? e % HASH_BITS : HASH_BITS-1-((-1-e) % HASH_BITS);
    x = ((x << e) & HASH_MODULUS) | x >> (HASH_BITS - e);
    x = x * sign;
    if (x == (Py_uhash_t)-1)
        x = (Py_uhash_t)-2;
    return (Py_hash_t)x;
}


// This is a "djb2" hash algorithm.
static inline Py_hash_t
unicode_to_hash(Py_UCS4 *str, Py_ssize_t len) {
    Py_UCS4* p = str;
    Py_UCS4* p_end = str + len;
    Py_hash_t hash = 5381;
    while (p < p_end) {
        hash = ((hash << 5) + hash) + *p++;
    }
    if (hash == -1) {
        return -2;
    }
    return hash;
}


static inline Py_hash_t
string_to_hash(char *str, Py_ssize_t len) {
    char* p = str;
    char* p_end = str + len;
    Py_hash_t hash = 5381;
    while (p < p_end) {
        hash = ((hash << 5) + hash) + *p++;
    }
    if (hash == -1) {
        return -2;
    }
    return hash;
}


//------------------------------------------------------------------------------
// the global int_cache is shared among all instances

static PyObject *int_cache = NULL;


// NOTE: this used to be a Py_ssize_t, which can be 32 bits on some machines and might easily overflow with a few very large indices. Using an explicit 64-bit int seems safer
static npy_int64 key_count_global = 0;


// Fill the int_cache up to size_needed with PyObject ints; `size` is not the key_count_global.
static int
int_cache_fill(Py_ssize_t size_needed)
{
    PyObject *item;
    if (!int_cache) {
        int_cache = PyList_New(0);
        if (!int_cache) {
            return -1;
        }
    }
    for (Py_ssize_t i = PyList_GET_SIZE(int_cache); i < size_needed; i++) {
        item = PyLong_FromSsize_t(i);
        if (!item) {
            return -1;
        }
        if (PyList_Append(int_cache, item)) {
            Py_DECREF(item);
            return -1;
        }
        Py_DECREF(item);
    }
    return 0;
}


// Given the current key_count_global, remove cache elements only if the key_count is less than the the current size of the int_cache.
void
int_cache_remove(Py_ssize_t key_count)
{
    if (!key_count) {
        Py_CLEAR(int_cache);
    }
    else if (key_count < PyList_GET_SIZE(int_cache)) {
        // del int_cache[key_count:]
        PyList_SetSlice(int_cache, key_count, PyList_GET_SIZE(int_cache), NULL);
    }
}


//------------------------------------------------------------------------------
// FrozenAutoMapIterator functions

typedef struct FAMIObject {
    PyObject_VAR_HEAD
    FAMObject *fam;
    PyArrayObject* keys_array;
    ViewKind kind;
    int reversed;
    Py_ssize_t index; // current index state, mutated in-place
} FAMIObject;


static void
fami_dealloc(FAMIObject *self)
{
    Py_DECREF(self->fam);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


static FAMIObject *
fami_iter(FAMIObject *self)
{
    Py_INCREF(self);
    return self;
}


// For a FAMI, Return appropriate PyObject for items, keys, and values. When values are needed they are retrieved from the int_cache. For consistency with NumPy array iteration, arrays use PyArray_ToScalar instead of PyArray_GETITEM.
static PyObject *
fami_iternext(FAMIObject *self)
{
    Py_ssize_t index;
    if (self->reversed) {
        index = self->fam->keys_size - ++self->index;
        if (index < 0) {
            return NULL;
        }
    }
    else {
        index = self->index++;
    }
    if (self->fam->keys_size <= index) {
        return NULL;
    }
    switch (self->kind) {
        case ITEMS: {
            if (self->fam->keys_array_type) {
                return PyTuple_Pack(
                    2,
                    PyArray_ToScalar(PyArray_GETPTR1(self->keys_array, index), self->keys_array),
                    PyList_GET_ITEM(int_cache, index)
                );
            }
            else {
                return PyTuple_Pack(
                    2,
                    PyList_GET_ITEM(self->fam->keys, index),
                    PyList_GET_ITEM(int_cache, index)
                );
            }
        }
        case KEYS: {
            if (self->fam->keys_array_type) {
                return PyArray_ToScalar(PyArray_GETPTR1(self->keys_array, index), self->keys_array);
            }
            else {
                PyObject* yield = PyList_GET_ITEM(self->fam->keys, index);
                Py_INCREF(yield);
                return yield;
            }
        }
        case VALUES: {
            PyObject *yield = PyList_GET_ITEM(int_cache, index);
            Py_INCREF(yield);
            return yield;
        }
    }
    Py_UNREACHABLE();
}


static PyObject *
fami___length_hint__(FAMIObject *self)
{
    Py_ssize_t len = Py_MAX(0, self->fam->keys_size - self->index);
    return PyLong_FromSsize_t(len);
}


static PyObject *fami_new(FAMObject *, ViewKind, int);


static PyObject *
fami___reversed__(FAMIObject *self)
{
    return fami_new(self->fam, self->kind, !self->reversed);
}


static PyMethodDef fami_methods[] = {
    {"__length_hint__", (PyCFunction)fami___length_hint__, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction)fami___reversed__, METH_NOARGS, NULL},
    {NULL},
};


static PyTypeObject FAMIType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_basicsize = sizeof(FAMIObject),
    .tp_dealloc = (destructor) fami_dealloc,
    .tp_iter = (getiterfunc) fami_iter,
    .tp_iternext = (iternextfunc) fami_iternext,
    .tp_methods = fami_methods,
    .tp_name = "automap.FrozenAutoMapIterator",
};


static PyObject *
fami_new(FAMObject *fam, ViewKind kind, int reversed)
{
    FAMIObject *fami = PyObject_New(FAMIObject, &FAMIType);
    if (!fami) {
        return NULL;
    }
    Py_INCREF(fam);
    fami->fam = fam;
    if (fam->keys_array_type) {
        fami->keys_array = (PyArrayObject *)fam->keys;
    }
    else {
        fami->keys_array = NULL;
    }
    fami->kind = kind;
    fami->reversed = reversed;
    fami->index = 0;
    return (PyObject *)fami;
}

//------------------------------------------------------------------------------
// FrozenAutoMapView functions

// A FAMVObject contains a reference to the FAM from which it was derived
typedef struct FAMVObject{
    PyObject_VAR_HEAD
    FAMObject *fam;
    ViewKind kind;
} FAMVObject;

# define FAMV_SET_OP(name, op)                            \
static PyObject *                                         \
name(PyObject *left, PyObject *right)                     \
{                                                         \
    left = PySet_New(left);                               \
    if (!left) {                                          \
        return NULL;                                      \
    }                                                     \
    right = PySet_New(right);                             \
    if (!right) {                                         \
        Py_DECREF(left);                                  \
        return NULL;                                      \
    }                                                     \
    PyObject *result = PyNumber_InPlace##op(left, right); \
    Py_DECREF(left);                                      \
    Py_DECREF(right);                                     \
    return result;                                        \
}


FAMV_SET_OP(famv_and, And)
FAMV_SET_OP(famv_or, Or)
FAMV_SET_OP(famv_subtract, Subtract)
FAMV_SET_OP(famv_xor, Xor)

# undef FAMV_SET_OP


static PyNumberMethods famv_as_number = {
    .nb_and = (binaryfunc) famv_and,
    .nb_or = (binaryfunc) famv_or,
    .nb_subtract = (binaryfunc) famv_subtract,
    .nb_xor = (binaryfunc) famv_xor,
};


static int fam_contains(FAMObject *, PyObject *);
static PyObject *famv_fami_new(FAMVObject *);


static int
famv_contains(FAMVObject *self, PyObject *other)
{
    if (self->kind == KEYS) {
        return fam_contains(self->fam, other);
    }
    PyObject *iterator = famv_fami_new(self);
    if (!iterator) {
        return -1;
    }
    int result = PySequence_Contains(iterator, other);
    Py_DECREF(iterator);
    return result;
}


static PySequenceMethods famv_as_sequence = {
    .sq_contains = (objobjproc) famv_contains,
};


static void
famv_dealloc(FAMVObject *self)
{
    Py_DECREF(self->fam);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject *
famv_fami_new(FAMVObject *self)
{
    return fami_new(self->fam, self->kind, 0);
}


static PyObject *
famv___length_hint__(FAMVObject *self)
{
    return PyLong_FromSsize_t(self->fam->keys_size);
}


static PyObject *
famv___reversed__(FAMVObject *self)
{
    return fami_new(self->fam, self->kind, 1);
}


static PyObject *
famv_isdisjoint(FAMVObject *self, PyObject *other)
{
    PyObject *intersection = famv_and((PyObject *)self, other);
    if (!intersection) {
        return NULL;
    }
    Py_ssize_t result = PySet_GET_SIZE(intersection);
    Py_DECREF(intersection);
    return PyBool_FromLong(result);
}


static PyObject *
famv_richcompare(FAMVObject *self, PyObject *other, int op)
{
    PyObject *left = PySet_New((PyObject *)self);
    if (!left) {
        return NULL;
    }
    PyObject *right = PySet_New(other);
    if (!right) {
        Py_DECREF(left);
        return NULL;
    }
    PyObject *result = PyObject_RichCompare(left, right, op);
    Py_DECREF(left);
    Py_DECREF(right);
    return result;
}


static PyMethodDef famv_methods[] = {
    {"__length_hint__", (PyCFunction) famv___length_hint__, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction) famv___reversed__, METH_NOARGS, NULL},
    {"isdisjoint", (PyCFunction) famv_isdisjoint, METH_O, NULL},
    {NULL},
};


static PyTypeObject FAMVType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_number = &famv_as_number,
    .tp_as_sequence = &famv_as_sequence,
    .tp_basicsize = sizeof(FAMVObject),
    .tp_dealloc = (destructor) famv_dealloc,
    .tp_iter = (getiterfunc) famv_fami_new,
    .tp_methods = famv_methods,
    .tp_name = "automap.FrozenAutoMapView",
    .tp_richcompare = (richcmpfunc) famv_richcompare,
};


static PyObject *
famv_new(FAMObject *fam, int kind)
{
    FAMVObject *famv = (FAMVObject *)PyObject_New(FAMVObject, &FAMVType);
    if (!famv) {
        return NULL;
    }
    famv->kind = kind;
    famv->fam = fam;
    Py_INCREF(fam);
    return (PyObject *)famv;
}

//------------------------------------------------------------------------------
// FrozenAutoMap functions

// Given a key and a computed hash, return the table_pos if that hash and key are found, or if not, the first table position that has not been assigned. Return -1 on error.
static Py_ssize_t
lookup_hash_obj(FAMObject *self, PyObject *key, Py_hash_t hash)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask;

    PyObject *guess = NULL;
    PyObject *keys = self->keys;
    int result = -1;
    Py_hash_t h = 0;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) { // Miss. Found a position that can be used for insertion.
                return table_pos;
            }
            if (h != hash) { // Collision.
                table_pos++;
                continue;
            }
            guess = PyList_GET_ITEM(keys, table[table_pos].keys_pos);
            if (guess == key) { // Hit. Object ID comparison
                return table_pos;
            }
            result = PyObject_RichCompareBool(guess, key, Py_EQ);
            if (result < 0) { // Error.
                return -1;
            }
            if (result) { // Hit.
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


static Py_ssize_t
lookup_hash_int(FAMObject *self, npy_int64 key, Py_hash_t hash)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask; // taking the modulo

    PyArrayObject *a = (PyArrayObject *)self->keys;
    npy_int64 k = 0;
    Py_hash_t h = 0;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) { // Miss. Position that can be used for insertion.
                return table_pos;
            }
            if (h != hash) {
                table_pos++;
                continue;
            }
            switch (self->keys_array_type) {
                case KAT_INT64:
                    k = *(npy_int64*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_INT32:
                    k = *(npy_int32*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_INT16:
                    k = *(npy_int16*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_INT8:
                    k = *(npy_int8*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                default:
                    return -1;
            }
            if (key == k) {
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


static Py_ssize_t
lookup_hash_uint(FAMObject *self, npy_uint64 key, Py_hash_t hash)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask;

    PyArrayObject *a = (PyArrayObject *)self->keys;
    npy_uint64 k = 0;
    Py_hash_t h = 0;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) {
                return table_pos;
            }
            if (h != hash) {
                table_pos++;
                continue;
            }
            switch (self->keys_array_type) {
                case KAT_UINT64:
                    k = *(npy_uint64*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_UINT32:
                    k = *(npy_uint32*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_UINT16:
                    k = *(npy_uint16*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_UINT8:
                    k = *(npy_uint8*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                default:
                    return -1;
            }
            if (key == k) {
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


static Py_ssize_t
lookup_hash_double(FAMObject *self, npy_double key, Py_hash_t hash)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask;

    PyArrayObject *a = (PyArrayObject *)self->keys;
    npy_double k = 0;
    Py_hash_t h = 0;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) {
                return table_pos;
            }
            if (h != hash) {
                table_pos++;
                continue;
            }
            switch (self->keys_array_type) {
                case KAT_FLOAT64:
                    k = *(npy_double*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_FLOAT32:
                    k = *(npy_float*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_FLOAT16:
                    k = npy_half_to_double(*(npy_half*)PyArray_GETPTR1(a, table[table_pos].keys_pos));
                    break;
                default:
                    return -1;
            }
            if (key == k) {
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


// Compare a passed Py_UCS4 array to stored keys. This does not use any dynamic memory. Returns -1 on error.
static Py_ssize_t
lookup_hash_unicode(
        FAMObject *self,
        Py_UCS4* key,
        Py_ssize_t key_size,
        Py_hash_t hash)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask;

    PyArrayObject *a = (PyArrayObject *)self->keys;
    // REVIEW: is this a new descr reference?
    Py_ssize_t dt_size = PyArray_DESCR(a)->elsize / sizeof(Py_UCS4);

    Py_hash_t h = 0;
    Py_UCS4* p_start = NULL;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) {
                return table_pos;
            }
            if (h != hash) {
                table_pos++;
                continue;
            }
            p_start = (Py_UCS4*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
            // memcmp returns 0 on match
            if (!memcmp(p_start, key, Py_MIN(key_size, dt_size))) {
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


// Compare a passed char array to stored keys. This does not use any dynamic memory. Returns -1 on error.
static Py_ssize_t
lookup_hash_string(
        FAMObject *self,
        char* key,
        Py_ssize_t key_size,
        Py_hash_t hash)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask;

    PyArrayObject *a = (PyArrayObject *)self->keys;
    Py_ssize_t dt_size = PyArray_DESCR(a)->elsize / sizeof(char);

    Py_hash_t h = 0;
    char* p_start = NULL;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) {
                return table_pos;
            }
            if (h != hash) {
                table_pos++;
                continue;
            }
            p_start = (char*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
            // memcmp returns 0 on match
            if (!memcmp(p_start, key, Py_MIN(key_size, dt_size))) {
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


static Py_ssize_t
lookup_int(FAMObject *self, PyObject* key) {
    npy_int64 v = 0;
    // NOTE: we handle PyArray Scalar Byte, Short, UByte, UShort with PyNumber_Check, below, saving four branches here
    if (PyArray_IsScalar(key, LongLong)) {
        v = (npy_int64)PyArrayScalar_VAL(key, LongLong);
    }
    else if (PyArray_IsScalar(key, Long)) {
        v = (npy_int64)PyArrayScalar_VAL(key, Long);
    }
    else if (PyLong_Check(key)) {
        v = PyLong_AsLongLong(key);
        if (v == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
    }
    else if (PyArray_IsScalar(key, Double)) {
        double dv = PyArrayScalar_VAL(key, Double);
        if (floor(dv) != dv) {
            return -1;
        }
        v = (npy_int64)dv;
    }
    else if (PyFloat_Check(key)) {
        double dv = PyFloat_AsDouble(key);
        if (dv == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
        v = (npy_int64)dv; // truncate to integer
        if (v != dv) {
            return -1;
        }
    }
    else if (PyArray_IsScalar(key, ULongLong)) {
        v = (npy_int64)PyArrayScalar_VAL(key, ULongLong);
    }
    else if (PyArray_IsScalar(key, ULong)) {
        v = (npy_int64)PyArrayScalar_VAL(key, ULong);
    }
    else if (PyArray_IsScalar(key, Int)) {
        v = (npy_int64)PyArrayScalar_VAL(key, Int);
    }
    else if (PyArray_IsScalar(key, UInt)) {
        v = (npy_int64)PyArrayScalar_VAL(key, UInt);
    }
    else if (PyArray_IsScalar(key, Float)) {
        double dv = (double)PyArrayScalar_VAL(key, Float);
        if (floor(dv) != dv) {
            return -1;
        }
        v = (npy_int64)dv;
    }
    else if (PyArray_IsScalar(key, Half)) {
        double dv = npy_half_to_double(PyArrayScalar_VAL(key, Half));
        if (floor(dv) != dv) {
            return -1;
        }
        v = (npy_int64)dv;
    }
    else if (PyBool_Check(key)) {
        v = PyObject_IsTrue(key);
    }
    else if (PyNumber_Check(key)) {
        // NOTE: this returns a Py_ssize_t, which might be 32 bit. This can be used for PyArray_Scalars <= ssize_t.
        v = (npy_int64)PyNumber_AsSsize_t(key, PyExc_OverflowError);
        if (v == -1 && PyErr_Occurred()) {
            return -1;
        }
    }
    else {
        return -1;
    }
    Py_hash_t hash = int_to_hash(v);
    return lookup_hash_int(self, v, hash);
}


static Py_ssize_t
lookup_uint(FAMObject *self, PyObject* key) {
    npy_uint64 v = 0;

    // NOTE: we handle PyArray Scalar Byte, Short, UByte, UShort with PyNumber_Check, below, saving four branches here
    if (PyArray_IsScalar(key, ULongLong)) {
        v = (npy_uint64)PyArrayScalar_VAL(key, ULongLong);
    }
    else if (PyArray_IsScalar(key, ULong)) {
        v = (npy_uint64)PyArrayScalar_VAL(key, ULong);
    }
    else if (PyArray_IsScalar(key, LongLong)) {
        npy_int64 si = (npy_int64)PyArrayScalar_VAL(key, LongLong);
        if (si < 0) {
            return -1;
        }
        v = (npy_uint64)si;
    }
    else if (PyArray_IsScalar(key, Long)) {
        npy_int64 si = (npy_int64)PyArrayScalar_VAL(key, Long);
        if (si < 0) {
            return -1;
        }
        v = (npy_uint64)si;
    }
    else if (PyLong_Check(key)) {
        v = PyLong_AsUnsignedLongLong(key);
        if (v == (unsigned long long)-1 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
    }
    else if (PyArray_IsScalar(key, Double)) {
        double dv = PyArrayScalar_VAL(key, Double);
        if (dv < 0 || floor(dv) != dv) {
            return -1;
        }
        v = (npy_uint64)dv;
    }
    else if (PyFloat_Check(key)) {
        double dv = PyFloat_AsDouble(key);
        if (dv == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
        if (dv < 0) {
            return -1;
        }
        v = (npy_uint64)dv; // truncate to integer
        if (v != dv) {
            return -1;
        }
    }
    else if (PyArray_IsScalar(key, Int)) {
        npy_int64 si = (npy_int64)PyArrayScalar_VAL(key, Int);
        if (si < 0) {
            return -1;
        }
        v = (npy_uint64)si;
    }
    else if (PyArray_IsScalar(key, UInt)) {
        v = (npy_uint64)PyArrayScalar_VAL(key, UInt);
    }
    else if (PyArray_IsScalar(key, Float)) {
        double dv = (double)PyArrayScalar_VAL(key, Float);
        if (dv < 0 || floor(dv) != dv) {
            return -1;
        }
        v = (npy_uint64)dv;
    }
    else if (PyArray_IsScalar(key, Half)) {
        double dv = npy_half_to_double(PyArrayScalar_VAL(key, Half));
        if (dv < 0 || floor(dv) != dv) {
            return -1;
        }
        v = (npy_uint64)dv;
    }
    else if (PyBool_Check(key)) {
        v = PyObject_IsTrue(key);
    }
    else if (PyNumber_Check(key)) {
        // NOTE: this returns a Py_ssize_t, which might be 32 bit. This can be used for PyArray_Scalars <= ssize_t.
        npy_int64 si = PyNumber_AsSsize_t(key, PyExc_OverflowError);
        if (si == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
        if (si < 0) {
            return -1;
        }
        v = (npy_uint64)si;
    }
    else {
        return -1;
    }
    Py_hash_t hash = uint_to_hash(v);
    return lookup_hash_uint(self, v, hash);
}


static Py_ssize_t
lookup_double(FAMObject *self, PyObject* key) {
        double v = 0;
        if (PyArray_IsScalar(key, Double)) {
            v = PyArrayScalar_VAL(key, Double);
        }
        else if (PyFloat_Check(key)) {
            v = PyFloat_AsDouble(key);
            if (v == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();
                return -1;
            }
        }
        else if (PyLong_Check(key)) {
            v = (double)PyLong_AsLongLong(key);
            if (v == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                return -1;
            }
        }
        // NOTE: we handle PyArray Scalar Byte, Short with PyNumber_Check, below, saving four branches here
        else if (PyArray_IsScalar(key, LongLong)) {
            v = (double)PyArrayScalar_VAL(key, LongLong);
        }
        else if (PyArray_IsScalar(key, Long)) {
            v = (double)PyArrayScalar_VAL(key, Long);
        }
        else if (PyArray_IsScalar(key, Int)) {
            v = (double)PyArrayScalar_VAL(key, Int);
        }
        else if (PyArray_IsScalar(key, ULongLong)) {
            v = (double)PyArrayScalar_VAL(key, ULongLong);
        }
        else if (PyArray_IsScalar(key, ULong)) {
            v = (double)PyArrayScalar_VAL(key, ULong);
        }
        else if (PyArray_IsScalar(key, UInt)) {
            v = (double)PyArrayScalar_VAL(key, UInt);
        }
        else if (PyArray_IsScalar(key, Float)) {
            v = (double)PyArrayScalar_VAL(key, Float);
        }
        else if (PyArray_IsScalar(key, Half)) {
            v = npy_half_to_double(PyArrayScalar_VAL(key, Half));
        }
        else if (PyBool_Check(key)) {
            v = PyObject_IsTrue(key);
        }
        else if (PyNumber_Check(key)) {
            // NOTE: this returns a Py_ssize_t, which might be 32 bit. This can be used for PyArray_Scalars <= ssize_t.
            npy_int64 si = PyNumber_AsSsize_t(key, PyExc_OverflowError);
            if (si == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                return -1;
            }
            v = (double)si;
        }
        else {
            return -1;
        }
        Py_hash_t hash = double_to_hash(v);
        return lookup_hash_double(self, v, hash);
}


static Py_ssize_t
lookup_unicode(FAMObject *self, PyObject* key) {
    // NOTE: while we can identify and use PyArray_IsScalar(key, Unicode), this did not improve performance and fails on Windows.
    if (!PyUnicode_Check(key)) {
        return -1;
    }
    PyArrayObject *a = (PyArrayObject *)self->keys;
    Py_ssize_t dt_size = PyArray_DESCR(a)->elsize / sizeof(Py_UCS4);
    // if the key_size is greater than the dtype size of the array, we know there cannot be a match
    Py_ssize_t k_size = PyUnicode_GetLength(key);
    if (k_size > dt_size) {
        return -1;
    }
    // The buffer will have dt_size + 1 storage. We copy a NULL character so do not have to clear the buffer, but instead can reuse it and still discover the lookup
    if (!PyUnicode_AsUCS4(key, self->key_buffer, dt_size+1, 1)) {
        return -1; // exception will be set
    }
    Py_hash_t hash = unicode_to_hash(self->key_buffer, k_size);
    return lookup_hash_unicode(self, self->key_buffer, k_size, hash);
}


static Py_ssize_t
lookup_string(FAMObject *self, PyObject* key) {
    if (!PyBytes_Check(key)) {
        return -1;
    }
    PyArrayObject *a = (PyArrayObject *)self->keys;
    Py_ssize_t dt_size = PyArray_DESCR(a)->elsize;
    Py_ssize_t k_size = PyBytes_GET_SIZE(key);
    if (k_size > dt_size) {
        return -1;
    }
    char* k = PyBytes_AS_STRING(key);
    Py_hash_t hash = string_to_hash(k, k_size);
    return lookup_hash_string(self, k, k_size, hash);
}


// Given a key as a PyObject, return the Py_ssize_t keys_pos value stored in the TableElement. Return -1 on key not found (without setting an exception) and -1 on error (with setting an exception).
static Py_ssize_t
lookup(FAMObject *self, PyObject *key) {
    Py_ssize_t table_pos = -1;

    switch (self->keys_array_type) {
        case KAT_INT64:
        case KAT_INT32:
        case KAT_INT16:
        case KAT_INT8:
            table_pos = lookup_int(self, key);
            break;
        case KAT_UINT64:
        case KAT_UINT32:
        case KAT_UINT16:
        case KAT_UINT8:
            table_pos = lookup_uint(self, key);
            break;
        case KAT_FLOAT64:
        case KAT_FLOAT32:
        case KAT_FLOAT16:
            table_pos = lookup_double(self, key);
            break;
        case KAT_UNICODE:
            table_pos = lookup_unicode(self, key);
            break;
        case KAT_STRING:
            table_pos = lookup_string(self, key);
            break;
        case KAT_LIST: {
            Py_hash_t hash = PyObject_Hash(key);
            if (hash == -1) {
                return -1;
            }
            table_pos = lookup_hash_obj(self, key, hash);
            break;
        }
    }
    // A -1 hash is an unused storage location
    if ((table_pos < 0) || (self->table[table_pos].hash == -1)) {
        return -1;
    }
    return self->table[table_pos].keys_pos;
}

// Insert a key_pos, hash pair into the table. Assumes table already has appropriate size. When inserting a new itme, `hash` is -1, forcing a fresh hash to be computed here. Return 0 on success, -1 on error.
static int
insert_obj(FAMObject *self, PyObject *key, Py_ssize_t keys_pos, Py_hash_t hash)
{
    if (hash == -1) {
        hash = PyObject_Hash(key);
        if (hash == -1) {
            return -1;
        }
    }
    // table position is not dependent on keys_pos
    Py_ssize_t table_pos = lookup_hash_obj(self, key, hash);

    if (table_pos < 0) {
        return -1;
    }
    // We expect, on insertion, to get back a table_pos that points to an unassigned hash value (-1); if we get anything else, we have found a match to an already-existing key, and thus raise a NonUniqueError error.
    if (self->table[table_pos].hash != -1) {
        PyErr_SetObject(NonUniqueError, key);
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash;
    return 0;
}


static int
insert_int(
        FAMObject *self,
        npy_int64 key,
        Py_ssize_t keys_pos,
        Py_hash_t hash)
{
    if (hash == -1) {
        hash = int_to_hash(key);
    }
    // table position is not dependent on keys_pos
    Py_ssize_t table_pos = lookup_hash_int(self, key, hash);

    if (table_pos < 0) {
        return -1;
    }
    if (self->table[table_pos].hash != -1) {
        PyErr_SetObject(NonUniqueError, PyLong_FromSsize_t(key));
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash; // key is the hash
    return 0;
}


static int
insert_uint(
        FAMObject *self,
        npy_uint64 key,
        Py_ssize_t keys_pos,
        Py_hash_t hash)
{
    if (hash == -1) {
        hash = uint_to_hash(key);
    }
    Py_ssize_t table_pos = lookup_hash_uint(self, key, hash);

    if (table_pos < 0) {
        return -1;
    }
    if (self->table[table_pos].hash != -1) {
        PyErr_SetObject(NonUniqueError, PyLong_FromSsize_t(key));
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash;
    return 0;
}


static int
insert_double(
        FAMObject *self,
        npy_double key,
        Py_ssize_t keys_pos,
        Py_hash_t hash)
{
    if (hash == -1) {
        hash = double_to_hash(key);
    }
    // table position is not dependent on keys_pos
    Py_ssize_t table_pos = lookup_hash_double(self, key, hash);

    if (table_pos < 0) {
        return -1;
    }
    if (self->table[table_pos].hash != -1) {
        PyErr_SetObject(NonUniqueError, PyFloat_FromDouble(key));
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash;
    return 0;
}


static int
insert_unicode(
        FAMObject *self,
        Py_UCS4* key,
        Py_ssize_t key_size,
        Py_ssize_t keys_pos,
        Py_hash_t hash)
{
    if (hash == -1) {
        hash = unicode_to_hash(key, key_size);
    }
    // table position is not dependent on keys_pos
    Py_ssize_t table_pos = lookup_hash_unicode(self, key, key_size, hash);
    if (table_pos < 0) {
        return -1;
    }
    if (self->table[table_pos].hash != -1) {
        PyErr_SetObject(NonUniqueError,
            PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, key, key_size));
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash;
    return 0;
}


static int
insert_string(
        FAMObject *self,
        char* key,
        Py_ssize_t key_size,
        Py_ssize_t keys_pos,
        Py_hash_t hash)
{
    if (hash == -1) {
        hash = string_to_hash(key, key_size);
    }
    // table position is not dependent on keys_pos
    Py_ssize_t table_pos = lookup_hash_string(self, key, key_size, hash);
    if (table_pos < 0) {
        return -1;
    }
    if (self->table[table_pos].hash != -1) {
        PyErr_SetObject(NonUniqueError,
            PyBytes_FromStringAndSize(key, key_size));
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash;
    return 0;
}


// Called in fam_new(), extend(), append(), with the size of observed keys. This table is updated only when append or extending. Only if there is an old table will keys be accessed Returns 0 on success, -1 on failure.
static int
grow_table(FAMObject *self, Py_ssize_t keys_size)
{
    // NOTE: this is the only place int_cache_fill is called; it is not called with key_count_global, but with the max value neede
    if (int_cache_fill(keys_size)) {
        return -1;
    }
    Py_ssize_t keys_load = keys_size / LOAD;
    Py_ssize_t size_old = self->table_size;
    if (keys_load < size_old) {
        return 0;
    }

    // get the next power of 2 greater than current keys_load
    Py_ssize_t size_new = 1;
    while (size_new <= keys_load) {
        size_new <<= 1;
    }
    // size_new > keys_load; we know that keys_load >= size_old, so size_new must be > size_old
    TableElement *table_old = self->table;
    TableElement *table_new = PyMem_New(TableElement, size_new + SCAN - 1);
    if (!table_new) {
        return -1;
    }

    // initialize all hash and keys_pos values to -1
    Py_ssize_t table_pos;
    for (table_pos = 0; table_pos < size_new + SCAN - 1; table_pos++) {
        table_new[table_pos].hash = -1;
        table_new[table_pos].keys_pos = -1;
    }
    self->table = table_new;
    self->table_size = size_new;

    // if we have an old table, move them into the new table
    if (size_old) {
        if (self->keys_array_type) {
            PyErr_SetString(PyExc_NotImplementedError, "Cannot grow table for array keys");
            goto restore;
        }

        Py_ssize_t i;
        Py_hash_t h;
        for (table_pos = 0; table_pos < size_old + SCAN - 1; table_pos++) {
            i = table_old[table_pos].keys_pos;
            h = table_old[table_pos].hash;
            if ((h != -1) && insert_obj(self, PyList_GET_ITEM(self->keys, i), i, h))
            {
                goto restore;
            }
        }
    }
    PyMem_Del(table_old);
    return 0;
restore:
    PyMem_Del(self->table);
    self->table = table_old;
    self->table_size = size_old;
    return -1;
}


// Given a new, possibly un-initialized FAMObject, copy attrs from self to new. Return 0 on success, -1 on error.
int
copy_to_new(PyTypeObject *cls, FAMObject *self, FAMObject *new)
{
    if (self->keys_array_type) {
        new->keys = self->keys;
        Py_INCREF(new->keys);
    }
    else {
        new->keys = PySequence_List(self->keys);
        if (!new->keys) {
            return -1;
        }
    }
    key_count_global += self->keys_size;

    new->table_size = self->table_size;
    new->keys_array_type = self->keys_array_type;
    new->keys_size = self->keys_size;

    new->key_buffer = NULL;
    if (new->keys_array_type == KAT_UNICODE) {
        PyArrayObject *a = (PyArrayObject *)new->keys;
        Py_ssize_t dt_size = PyArray_DESCR(a)->elsize / sizeof(Py_UCS4);
        new->key_buffer = (Py_UCS4*)PyMem_Malloc((dt_size+1) * sizeof(Py_UCS4));
    }

    Py_ssize_t table_size_alloc = new->table_size + SCAN - 1;
    new->table = PyMem_New(TableElement, table_size_alloc);
    if (!new->table) {
        // Py_DECREF(new->keys); // assume this will get cleaned up
        return -1;
    }
    memcpy(new->table, self->table, table_size_alloc * sizeof(TableElement));
    return 0;
}


static PyObject *
fam_new(PyTypeObject *cls, PyObject *args, PyObject *kwargs);


// Create a copy of self. Used in `fam_or()`. Returns a new FAMObject on success, NULL on error.
static FAMObject *
copy(PyTypeObject *cls, FAMObject *self)
{
    if (!PyType_IsSubtype(cls, &AMType) && !PyObject_TypeCheck(self, &AMType)) {
        Py_INCREF(self);
        return self;
    }
    // fam_new to allocate and full struct attrs
    FAMObject *new = (FAMObject*)fam_new(cls, NULL, NULL);
    if (!new) {
        return NULL;
    }
    if (copy_to_new(cls, self, new)) {
        Py_DECREF(new); // assume this will decref any partially set attrs of new
    }
    return new;
}



// Returns -1 on error, 0 on success.
static int
extend(FAMObject *self, PyObject *keys)
{
    if (self->keys_array_type) {
        PyErr_SetString(PyExc_NotImplementedError, "Not supported for array keys");
        return -1;
    }
    // this should fail for self->keys types that are not a list
    keys = PySequence_Fast(keys, "expected an iterable of keys");
    if (!keys) {
        return -1;
    }
    Py_ssize_t size_extend = PySequence_Fast_GET_SIZE(keys);
    key_count_global += size_extend;
    self->keys_size += size_extend;

    if (grow_table(self, self->keys_size)) {
        Py_DECREF(keys);
        return -1;
    }

    PyObject **keys_fi = PySequence_Fast_ITEMS(keys);

    for (Py_ssize_t index = 0; index < size_extend; index++) {
        // get the new keys_size after each append
        if (insert_obj(self, keys_fi[index], PyList_GET_SIZE(self->keys), -1) ||
            PyList_Append(self->keys, keys_fi[index]))
        {
            Py_DECREF(keys);
            return -1;
        }
    }
    Py_DECREF(keys);
    return 0;
}


// Returns -1 on error, 0 on success.
static int
append(FAMObject *self, PyObject *key)
{
    if (self->keys_array_type) {
        PyErr_SetString(PyExc_NotImplementedError, "Not supported for array keys");
        return -1;
    }
    key_count_global++;
    self->keys_size++;

    if (grow_table(self, self->keys_size)) {
        return -1;
    }
    // keys_size is already incremented; provide last index
    if (insert_obj(self, key, self->keys_size - 1, -1) ||
        PyList_Append(self->keys, key))
    {
        return -1;
    }
    return 0;
}


static Py_ssize_t
fam_length(FAMObject *self)
{
    return self->keys_size;
}


// Given a key for a FAM, return the Python integer (via the int_cache) associated with that key. Utility function used in both fam_subscript() and fam_get()
static PyObject *
get(FAMObject *self, PyObject *key, PyObject *missing) {
    Py_ssize_t keys_pos = lookup(self, key);
    if (keys_pos < 0) {
        if (PyErr_Occurred()) {
            return NULL;
        }
        if (missing) {
            Py_INCREF(missing);
            return missing;
        }
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }
    // use a C-integer to fetch the Python integer
    PyObject *index = PyList_GET_ITEM(int_cache, keys_pos);
    Py_INCREF(index);
    return index;
}


static PyObject *
fam_subscript(FAMObject *self, PyObject *key)
{
    return get(self, key, NULL);
}


static PyMappingMethods fam_as_mapping = {
    .mp_length = (lenfunc) fam_length,
    .mp_subscript = (binaryfunc) fam_subscript,
};


static PyObject *
fam_or(PyObject *left, PyObject *right)
{
    if (!PyObject_TypeCheck(left, &FAMType) ||
        !PyObject_TypeCheck(right, &FAMType)
    ) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    FAMObject *updated = copy(Py_TYPE(left), (FAMObject *)left);
    if (!updated) {
        return NULL;
    }
    if (extend(updated, ((FAMObject *)right)->keys)) {
        Py_DECREF(updated);
        return NULL;
    }
    return (PyObject *)updated;
}


static PyNumberMethods fam_as_number = {
    .nb_or = (binaryfunc) fam_or,
};


static int
fam_contains(FAMObject *self, PyObject *key)
{
    if (lookup(self, key) < 0) {
        if (PyErr_Occurred()) {
            return -1;
        }
        return 0;
    }
    return 1;
}


static PySequenceMethods fam_as_sequence = {
    .sq_contains = (objobjproc) fam_contains,
};


static void
fam_dealloc(FAMObject *self)
{
    if (self->table) {
        PyMem_Free(self->table);
    }
    if (self->key_buffer) {
        PyMem_Free(self->key_buffer);
    }
    if (self->keys) {
        Py_DECREF(self->keys);
    }

    key_count_global -= self->keys_size;

    Py_TYPE(self)->tp_free((PyObject *)self);
    int_cache_remove(key_count_global);
}


// Return a hash integer for an entire FAM by combining all stored hashes
static Py_hash_t
fam_hash(FAMObject *self)
{
    Py_hash_t hash = 0;
    for (Py_ssize_t i = 0; i < self->table_size; i++) {
        hash = hash * 3 + self->table[i].hash;
    }
    if (hash == -1) { // most not return -1
        return 0;
    }
    return hash;
}


static PyObject *
fam_iter(FAMObject *self)
{
    return fami_new(self, KEYS, 0);
}


static PyObject *
fam___getnewargs__(FAMObject *self)
{
    return PyTuple_Pack(1, self->keys);
}


static PyObject *
fam___reversed__(FAMObject *self)
{
    return fami_new(self, KEYS, 1);
}


static PyObject *
fam___sizeof__(FAMObject *self)
{
    PyObject *listsizeof = PyObject_CallMethod(self->keys, "__sizeof__", NULL);
    if (!listsizeof) {
        return NULL;
    }
    Py_ssize_t listbytes = PyLong_AsSsize_t(listsizeof);
    Py_DECREF(listsizeof);
    if (listbytes == -1 && PyErr_Occurred()) {
        return NULL;
    }
    return PyLong_FromSsize_t(
        Py_TYPE(self)->tp_basicsize
        + listbytes
        + (self->table_size + SCAN - 1) * sizeof(TableElement)
    );
}


static PyObject *
fam_get(FAMObject *self, PyObject *args)
{
    PyObject *key, *missing = Py_None;
    if (!PyArg_UnpackTuple(args, Py_TYPE(self)->tp_name, 1, 2, &key, &missing))
    {
        return NULL;
    }
    return get(self, key, missing);
}


static PyObject *
fam_items(FAMObject *self)
{
    return famv_new(self, ITEMS);
}


static PyObject *
fam_keys(FAMObject *self)
{
    return famv_new(self, KEYS);
}


static PyObject *
fam_values(FAMObject *self)
{
    return famv_new(self, VALUES);
}


static PyObject *
fam_new(PyTypeObject *cls, PyObject *args, PyObject *kwargs)
{
    // NOTE: The original fam_new used to be able to provide a same reference back if a fam was in the args; this is tricky now that we have fam_init
    FAMObject *self = (FAMObject *)cls->tp_alloc(cls, 0);
    if (!self) {
        return NULL;
    }
    self->table = NULL;
    self->keys = NULL;
    self->key_buffer = NULL;
    self->keys_size = 0;

    return (PyObject*)self;
}


// This macro can be used with integer and floating point NumPy types, given an `npy_type` and a specialized `insert_func`. Uses context of `fam_init` to get `fam`, `contiguous`, `a`, `keys_size`, and `i`. An optional `pre_insert` function can be supplied to transform extracted values before calling the appropriate insert function.
# define INSERT_SCALARS(npy_type, insert_func, pre_insert)    \
if (contiguous) {                                             \
    npy_type* b = (npy_type*)PyArray_DATA(a);                 \
    npy_type* b_end = b + keys_size;                          \
    while (b < b_end) {                                       \
        if (insert_func(fam, pre_insert(*b), i, -1)) {        \
            goto error;                                       \
        }                                                     \
        b++;                                                  \
        i++;                                                  \
    }                                                         \
}                                                             \
else {                                                        \
    for (; i < keys_size; i++) {                              \
        if (insert_func(fam,                                  \
                pre_insert(*(npy_type*)PyArray_GETPTR1(a, i)),\
                i,                                            \
                -1)) {                                        \
            goto error;                                       \
        }                                                     \
    }                                                         \
}                                                             \


// This macro is for inserting flexible-sized types, Unicode (Py_UCS4) or strings (char). Uses context of `fam_init`.
# define INSERT_FLEXIBLE(char_type, insert_func, get_end_func) \
char_type* p = NULL;                                           \
if (contiguous) {                                              \
    char_type *b = (char_type*)PyArray_DATA(a);                \
    char_type *b_end = b + keys_size * dt_size;                \
    while (b < b_end) {                                        \
        p = get_end_func(b, dt_size);                          \
        if (insert_func(fam, b, p-b, i, -1)) {                 \
            goto error;                                        \
        }                                                      \
        b += dt_size;                                          \
        i++;                                                   \
    }                                                          \
}                                                              \
else {                                                         \
    for (; i < keys_size; i++) {                               \
        char_type* v = (char_type*)PyArray_GETPTR1(a, i);      \
        p = get_end_func(v, dt_size);                          \
        if (insert_func(fam, v, p-v, i, -1)) {                 \
            goto error;                                        \
        }                                                      \
    }                                                          \
}                                                              \


// Initialize an allocated FAMObject. Returns 0 on success, -1 on error.
int
fam_init(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyTypeObject* cls = Py_TYPE(self); // borrowed ref
    const char *name = cls->tp_name;
    FAMObject* fam = (FAMObject*)self;

    if (kwargs) {
        PyErr_Format(PyExc_TypeError, "%s takes no keyword arguments", name);
        return -1;
    }

    int keys_array_type = KAT_LIST; // default, will override if necessary

    PyObject *keys = NULL;
    Py_ssize_t keys_size = 0;

    if (!PyArg_UnpackTuple(args, name, 0, 1, &keys)) {
        return -1;
    }

    if (!keys) {
        keys = PyList_New(0);
    }
    else if (PyObject_TypeCheck(keys, &FAMType)) {
        // Use `keys` as old, `self` as new, and fill from old to new. This returns the same error codes as this function.
        return copy_to_new(cls, (FAMObject*)keys, fam);
    }
    else if (PyArray_Check(keys)) {
        PyArrayObject *a = (PyArrayObject *)keys;
        if (PyArray_NDIM(a) != 1) {
            PyErr_SetString(PyExc_TypeError, "Arrays must be 1-dimensional");
            return -1;
        }
        int array_t = PyArray_TYPE(a);
        if (cls != &AMType &&
                (PyTypeNum_ISINTEGER(array_t) // signed and unsigned
                || PyTypeNum_ISFLOAT(array_t)
                || PyTypeNum_ISFLEXIBLE(array_t))
            ){
            if ((PyArray_FLAGS(a) & NPY_ARRAY_WRITEABLE)) {
                PyErr_Format(PyExc_TypeError, "Arrays must be immutable when given to a %s", name);
                return -1;
            }
            keys_array_type = at_to_kat(array_t);
            Py_INCREF(keys);
        }
        else { // if an AutoMap or an array that we do not custom-hash, we create a list
            if (array_t == NPY_DATETIME || array_t == NPY_TIMEDELTA){
                keys = PySequence_List(keys); // force scalars
            }
            else {
                keys = PyArray_ToList(a); // converts to objs
            }
        }
        keys_size = PyArray_SIZE(a);
    }
    else { // assume an arbitrary iterable
        keys = PySequence_List(keys);
        keys_size = PyList_GET_SIZE(keys);
    }

    if (!keys) {
        return -1;
    }

    fam->keys = keys;
    fam->keys_array_type = keys_array_type;
    fam->keys_size = keys_size;
    fam->key_buffer = NULL;
    key_count_global += keys_size;

    // NOTE: on itialization, grow_table() does not use keys
    if (grow_table(fam, keys_size)) {
        // assume `fam->keys` will be decrefed by the caller
        return -1;
    }
    Py_ssize_t i = 0;
    if (keys_array_type) {
        PyArrayObject *a = (PyArrayObject *)fam->keys;
        int contiguous = PyArray_IS_C_CONTIGUOUS(a);
        switch (keys_array_type) {
            case KAT_INT64:
                INSERT_SCALARS(npy_int64, insert_int,);
                break;
            case KAT_INT32:
                INSERT_SCALARS(npy_int32, insert_int,);
                break;
            case KAT_INT16:
                INSERT_SCALARS(npy_int16, insert_int,);
                break;
            case KAT_INT8:
                INSERT_SCALARS(npy_int8, insert_int,);
                break;
            case KAT_UINT64:
                INSERT_SCALARS(npy_uint64, insert_uint,);
                break;
            case KAT_UINT32:
                INSERT_SCALARS(npy_uint32, insert_uint,);
                break;
            case KAT_UINT16:
                INSERT_SCALARS(npy_uint16, insert_uint,);
                break;
            case KAT_UINT8:
                INSERT_SCALARS(npy_uint8, insert_uint,);
                break;
            case KAT_FLOAT64:
                INSERT_SCALARS(npy_double, insert_double,);
                break;
            case KAT_FLOAT32:
                INSERT_SCALARS(npy_float, insert_double,);
                break;
            case KAT_FLOAT16:
                INSERT_SCALARS(npy_half, insert_double, npy_half_to_double);
                break;
            case KAT_UNICODE: {
                // Over allocate buffer by 1 so there is room for null at end. This buffer is only used in lookup();
                Py_ssize_t dt_size = PyArray_DESCR(a)->elsize / sizeof(Py_UCS4);
                fam->key_buffer = (Py_UCS4*)PyMem_Malloc((dt_size+1) * sizeof(Py_UCS4));
                INSERT_FLEXIBLE(Py_UCS4, insert_unicode, ucs4_get_end_p);
                break;
            }
            case KAT_STRING: {
                Py_ssize_t dt_size = PyArray_DESCR(a)->elsize;
                INSERT_FLEXIBLE(char, insert_string, char_get_end_p);
                break;
            }
        }
    }
    else {
        for (; i < keys_size; i++) {
            if (insert_obj(fam, PyList_GET_ITEM(keys, i), i, -1)) {
                goto error;
            }
        }
    }
    return 0;
error:
    return -1;
}


# undef INSERT_SCALARS
# undef INSERT_FLEXIBLE


static PyObject *
fam_repr(FAMObject *self)
{
    return PyUnicode_FromFormat("%s(%R)", Py_TYPE(self)->tp_name, self->keys);
}


static PyObject *
fam_richcompare(FAMObject *self, PyObject *other, int op)
{
    if (!PyObject_TypeCheck(other, &FAMType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return PyObject_RichCompare(self->keys, ((FAMObject *)other)->keys, op);
}


static PyObject*
fam___getstate__(FAMObject *self)
{
    PyObject* state = PyTuple_Pack(1, self->keys);
    return state;
}


// State returned here is a tuple of keys, suitable for usage as an `args` argument.
static PyObject*
fam___setstate__(FAMObject *self, PyObject *state)
{
    if (!PyTuple_CheckExact(state) || !PyTuple_GET_SIZE(state)) {
        PyErr_SetString(PyExc_ValueError, "Unexpected pickled object.");
        return NULL;
    }
    PyObject *keys = PyTuple_GetItem(state, 0);
    if (PyArray_Check(keys)) {
        // if we an array, make it immutable
        PyArray_CLEARFLAGS((PyArrayObject*)keys, NPY_ARRAY_WRITEABLE);
    }
    fam_init((PyObject*)self, state, NULL);
    Py_RETURN_NONE;
}


static PyMethodDef fam_methods[] = {
    {"__getnewargs__", (PyCFunction) fam___getnewargs__, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction) fam___reversed__, METH_NOARGS, NULL},
    {"__sizeof__", (PyCFunction) fam___sizeof__, METH_NOARGS, NULL},
    {"__getstate__", (PyCFunction) fam___getstate__, METH_NOARGS, NULL},
    {"__setstate__", (PyCFunction) fam___setstate__, METH_O, NULL},
    {"get", (PyCFunction) fam_get, METH_VARARGS, NULL},
    {"items", (PyCFunction) fam_items, METH_NOARGS, NULL},
    {"keys", (PyCFunction) fam_keys, METH_NOARGS, NULL},
    {"values", (PyCFunction) fam_values, METH_NOARGS, NULL},
    {NULL},
};


static PyTypeObject FAMType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_mapping = &fam_as_mapping,
    .tp_as_number = &fam_as_number,
    .tp_as_sequence = &fam_as_sequence,
    .tp_basicsize = sizeof(FAMObject),
    .tp_dealloc = (destructor) fam_dealloc,
    .tp_doc = "An immutable auto-incremented integer-valued mapping.",
    .tp_hash = (hashfunc) fam_hash,
    .tp_iter = (getiterfunc) fam_iter,
    .tp_methods = fam_methods,
    .tp_name = "automap.FrozenAutoMap",
    .tp_new = fam_new,
    .tp_init = fam_init,
    .tp_repr = (reprfunc) fam_repr,
    .tp_richcompare = (richcmpfunc) fam_richcompare,
};


//------------------------------------------------------------------------------
// AutoMap subclass

static PyObject *
am_inplace_or(FAMObject *self, PyObject *other)
{
    if (PyObject_TypeCheck(other, &FAMType)) {
        other = ((FAMObject *)other)->keys;
    }
    if (extend(self, other)) {
        return NULL;
    }
    Py_INCREF(self);
    return (PyObject *)self;
}


static PyNumberMethods am_as_number = {
    .nb_inplace_or = (binaryfunc) am_inplace_or,
};


static PyObject *
am_add(FAMObject *self, PyObject *other)
{
    if (append(self, other)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject *
am_update(FAMObject *self, PyObject *other)
{
    if (PyObject_TypeCheck(other, &FAMType)) {
        other = ((FAMObject *)other)->keys;
    }
    if (extend(self, other)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyMethodDef am_methods[] = {
    {"add", (PyCFunction) am_add, METH_O, NULL},
    {"update", (PyCFunction) am_update, METH_O, NULL},
    {NULL},
};

static PyTypeObject AMType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_number = &am_as_number,
    .tp_base = &FAMType,
    .tp_doc = "A grow-only autoincremented integer-valued mapping.",
    .tp_methods = am_methods,
    .tp_name = "automap.AutoMap",
    .tp_richcompare = (richcmpfunc) fam_richcompare,
};


//------------------------------------------------------------------------------
// module definition

static struct PyModuleDef automap_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_doc = "High-performance autoincremented integer-valued mappings.",
    .m_name = "automap",
    .m_size = -1,
};


PyObject *
PyInit_automap(void)
{
    import_array();

    NonUniqueError = PyErr_NewExceptionWithDoc(
            "automap.NonUniqueError",
            "ValueError for non-unique values.",
            PyExc_ValueError,
            NULL);
    if (NonUniqueError == NULL) {
        return NULL;
    }

    PyObject *automap = PyModule_Create(&automap_module);
    if (
        !automap
        || PyType_Ready(&AMType)
        || PyType_Ready(&FAMIType)
        || PyType_Ready(&FAMVType)
        || PyType_Ready(&FAMType)
        || PyModule_AddObject(automap, "AutoMap", (PyObject *)&AMType)
        || PyModule_AddObject(automap, "FrozenAutoMap", (PyObject *)&FAMType)
        || PyModule_AddObject(automap, "NonUniqueError", NonUniqueError)
    ) {
        Py_XDECREF(automap);
        return NULL;
    }
    return automap;
}
