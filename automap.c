// TODO: More tests.
// TODO: Rewrite performance tests using pyperf.
// TODO: Group similar functionality.
// TODO: Check refcounts when calling into hash and comparison functions.
// TODO: Check allocation and cleanup.
// TODO: Subinterpreter support.
// TODO: Docstrings and stubs.
// TODO: GC support.
// TODO: More comments.


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

# define PY_SSIZE_T_CLEAN
# include "Python.h"

# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"
# include "numpy/arrayscalars.h"

// PyPy doesn't define Py_UNREACHABLE():

# ifndef Py_UNREACHABLE
# define Py_UNREACHABLE() Py_FatalError("https://xkcd.com/2200")
# endif

// Experimentation shows that these values work well:

# define LOAD 0.9
# define SCAN 16

//------------------------------------------------------------------------------
// Common

static PyTypeObject AMType;
static PyTypeObject FAMIType;
static PyTypeObject FAMVType;
static PyTypeObject FAMType;
static PyObject *NonUniqueError;


// The main storage "table" is an array of TableElement
typedef struct {
    Py_ssize_t keys_pos;
    Py_hash_t hash;
} TableElement;

typedef struct {
    TableElement *table;
    Py_ssize_t table_size;
} HashTable;

typedef struct {
    PyObject_VAR_HEAD
    Py_ssize_t table_size;
    TableElement *table;    // an array of TableElement structs
    PyObject *keys;  // I want this to be an immutable NumPy array
} FAMObject;

typedef enum {
    ITEMS,
    KEYS,
    VALUES,
} ViewKind;


//------------------------------------------------------------------------------
// the global int_cache is shared among all instances

static PyObject *int_cache = NULL;
static Py_ssize_t int_cache_count = 0;

// Fill the int_cache up to size with PyObject ints.
static int
int_cache_fill(Py_ssize_t size)
{
    PyObject *item;
    if (!int_cache) {
        int_cache = PyList_New(0);
        if (!int_cache) {
            return -1;
        }
    }
    for (Py_ssize_t index = PyList_GET_SIZE(int_cache); index < size; index++) {
        item = PyLong_FromSsize_t(index);
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

// Conform the size of the int_cache to the requested size
void
int_cache_remove(Py_ssize_t size)
{
    if (!size) {
        Py_CLEAR(int_cache);
    }
    else if (size < PyList_GET_SIZE(int_cache)) {
        // del int_cache[size:]
        PyList_SetSlice(int_cache, size, PyList_GET_SIZE(int_cache), NULL);
    }
}


//------------------------------------------------------------------------------
// FrozenAutoMapIterator objects

typedef struct {
    PyObject_VAR_HEAD
    FAMObject *fam;
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


// For a FAMI, Return appropriate PyObject for items, keys, and values. When values are needed they are retrieved from the int_cache
static PyObject *
fami_iternext(FAMIObject *self)
{
    Py_ssize_t index;
    if (self->reversed) {
        index = PyList_GET_SIZE(self->fam->keys) - ++self->index;
        if (index < 0) {
            return NULL;
        }
    }
    else {
        index = self->index++;
    }
    if (PyList_GET_SIZE(self->fam->keys) <= index) {
        return NULL;
    }
    switch (self->kind) {
        case ITEMS: {
            return PyTuple_Pack(
                2,
                PyList_GET_ITEM(self->fam->keys, index),
                PyList_GET_ITEM(int_cache, index)
            );
        }
        case KEYS: {
            PyObject *yield = PyList_GET_ITEM(self->fam->keys, index);
            Py_INCREF(yield);
            return yield;
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
    Py_ssize_t len = Py_MAX(0, PyList_GET_SIZE(self->fam->keys) - self->index);
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
    fami->kind = kind;
    fami->reversed = reversed;
    fami->index = 0;
    return (PyObject *)fami;
}

//------------------------------------------------------------------------------
// FrozenAutoMapView objects


// A FAMVObject contains a reference to the FAM from which it was derived
typedef struct {
    PyObject_VAR_HEAD
    FAMObject *fam;
    ViewKind kind;
} FAMVObject;

# define FAMV_SET_OP(name, op)                                 \
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
    // Py_DECREF(other); // previously we did this, which would segfault
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
    return PyLong_FromSsize_t(PyList_GET_SIZE(self->fam->keys));
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

// Given a key and a computed hash, return the table_pos if that hash and key are found
static Py_ssize_t
lookup_hash(FAMObject *self, PyObject *key, Py_hash_t hash)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);

    PyObject **keys = PySequence_Fast_ITEMS(self->keys); // returns underlying array of PyObject pointers
    Py_ssize_t table_pos = hash & mask; // taking the modulo

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            Py_hash_t h = table[table_pos].hash;
            if (h == -1) {
                // Miss.
                return table_pos;
            }
            if (h != hash) {
                // Collision.
                table_pos++;
                continue;
            }
            PyObject *guess = keys[table[table_pos].keys_pos];

            // try object id compare first
            if (guess == key) {
                // Hit.
                return table_pos;
            }
            int result = PyObject_RichCompareBool(guess, key, Py_EQ);
            if (result < 0) {
                // Error.
                return -1;
            }
            if (result) {
                // Hit.
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


// Given a key, return the Py_ssize_t keys position value stored in the TableElement
static Py_ssize_t
lookup(FAMObject *self, PyObject *key) {
    Py_hash_t hash = PyObject_Hash(key);
    if (hash == -1) {
        return -1;
    }
    Py_ssize_t table_pos = lookup_hash(self, key, hash);

    // why would the table have a -1 as a hash at this index
    if ((table_pos < 0) || (self->table[table_pos].hash == -1)) {
        return -1;
    }
    return self->table[table_pos].keys_pos;
}

// Insert a key_pos, hash pair into the table. Assumes table already has appropriate size. Return 0 on success, -1 on error.
static int
insert(FAMObject *self, PyObject *key, Py_ssize_t keys_pos, Py_hash_t hash)
{
    if (hash == -1) {
        hash = PyObject_Hash(key);
        if (hash == -1) {
            return -1;
        }
    }
    Py_ssize_t table_pos = lookup_hash(self, key, hash);
    if (table_pos < 0) {
        return -1;
    }
    if (self->table[table_pos].hash != -1) {
        PyErr_SetObject(NonUniqueError, key);
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash;
    return 0;
}


// Returns 0 on succes, -1 on failure.
static int
grow_table(FAMObject *self, Py_ssize_t needed)
{
    if (int_cache_fill(needed)) {
        return -1;
    }
    Py_ssize_t size_old = self->table_size;
    Py_ssize_t size_new = 1;
    needed /= LOAD;

    while (size_new <= needed) {
        size_new <<= 1;
    }
    if (size_new <= size_old) {
        return 0;
    }

    TableElement *table_old = self->table;
    TableElement *table_new = PyMem_New(TableElement, size_new + SCAN - 1);
    if (!table_new) {
        return -1;
    }

    Py_ssize_t table_pos;
    for (table_pos = 0; table_pos < size_new + SCAN - 1; table_pos++) {
        table_new[table_pos].hash = -1;
        table_new[table_pos].keys_pos = -1;
    }
    self->table = table_new;
    self->table_size = size_new;

    // if we have an old table, move them into the new table
    if (size_old) {
        for (table_pos = 0; table_pos < size_old + SCAN - 1; table_pos++) {
            if ((table_old[table_pos].hash != -1) &&
                insert(self,
                        PyList_GET_ITEM(self->keys, table_old[table_pos].keys_pos),
                        table_old[table_pos].keys_pos,
                        table_old[table_pos].hash))
            {
                // on error, delete the new table and re-assign the old
                PyMem_Del(self->table);
                self->table = table_old;
                self->table_size = size_old;
                return -1;
            }
        }
    }
    PyMem_Del(table_old);
    return 0;
}


static FAMObject *
copy(PyTypeObject *cls, FAMObject *self)
{
    if (!PyType_IsSubtype(cls, &AMType) && !PyObject_TypeCheck(self, &AMType)) {
        Py_INCREF(self);
        return self;
    }
    PyObject *keys = PySequence_List(self->keys);
    if (!keys) {
        return NULL;
    }
    FAMObject *new = (FAMObject *)cls->tp_alloc(cls, 0);
    if (!new) {
        Py_DECREF(keys);
        return NULL;
    }
    int_cache_count += PyList_GET_SIZE(keys);
    new->keys = keys;
    new->table_size = self->table_size;

    Py_ssize_t table_size_alloc = new->table_size + SCAN - 1;
    new->table = PyMem_New(TableElement, table_size_alloc);
    if (!new->table) {
        Py_DECREF(new);
        return NULL;
    }
    memcpy(new->table, self->table, table_size_alloc * sizeof(TableElement));
    return new;
}


static int
extend(FAMObject *self, PyObject *keys)
{
    keys = PySequence_Fast(keys, "expected an iterable of keys");
    if (!keys) {
        return -1;
    }
    Py_ssize_t size_extend = PySequence_Fast_GET_SIZE(keys);
    int_cache_count += size_extend;

    if (grow_table(self, PyList_GET_SIZE(self->keys) + size_extend)) {
        Py_DECREF(keys);
        return -1;
    }
    PyObject **keys_array = PySequence_Fast_ITEMS(keys);

    for (Py_ssize_t index = 0; index < size_extend; index++) {
        if (insert(self, keys_array[index], PyList_GET_SIZE(self->keys), -1) ||
            PyList_Append(self->keys, keys_array[index]))
        {
            Py_DECREF(keys);
            return -1;
        }
    }
    Py_DECREF(keys);
    return 0;
}


static int
append(FAMObject *self, PyObject *key)
{
    int_cache_count++;
    if (grow_table(self, PyList_GET_SIZE(self->keys) + 1)) {
        return -1;
    }
    if (insert(self, key, PyList_GET_SIZE(self->keys), -1) ||
        PyList_Append(self->keys, key))
    {
        return -1;
    }
    return 0;
}


static Py_ssize_t
fam_length(FAMObject *self)
{
    return PyList_GET_SIZE(self->keys);
}


// Given a key for a FAM, return the Python integer (via the int_cache) associated with that key. Utility function used in both fam_subscript() and fam_get()
static PyObject *
get(FAMObject *self, PyObject *key, PyObject *missing) {
    Py_ssize_t result = lookup(self, key);
    if (result < 0) {
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
    PyObject *index = PyList_GET_ITEM(int_cache, result);
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
    PyMem_Del(self->table);
    int_cache_count -= PyList_GET_SIZE(self->keys);
    Py_DECREF(self->keys);
    Py_TYPE(self)->tp_free((PyObject *)self);
    int_cache_remove(int_cache_count);
}


// Return a hash integer for an entire FAM by combining all stored hashes
static Py_hash_t
fam_hash(FAMObject *self)
{
    Py_hash_t hash = 0;
    for (Py_ssize_t i = 0; i < self->table_size; i++) {
        hash = hash * 3 + self->table[i].hash;
    }
    if (hash == -1) {
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
    const char *name = cls->tp_name;
    if (kwargs) {
        PyErr_Format(PyExc_TypeError, "%s takes no keyword arguments", name);
        return NULL;
    }

    PyObject *keys = NULL;
    if (!PyArg_UnpackTuple(args, name, 0, 1, &keys)) {
        return NULL;
    }

    if (!keys) {
        keys = PyList_New(0);
    }
    else if (PyObject_TypeCheck(keys, &FAMType)) {
        return (PyObject *)copy(cls, (FAMObject *)keys);
    }
    else {
        keys = PySequence_List(keys);
    }

    if (!keys) {
        return NULL;
    }

    FAMObject *self = (FAMObject *)cls->tp_alloc(cls, 0);
    if (!self) {
        Py_DECREF(keys);
        return NULL;
    }

    self->keys = keys;
    int_cache_count += PyList_GET_SIZE(keys);
    if (grow_table(self, PyList_GET_SIZE(keys))) {
        Py_DECREF(self);
        return NULL;
    }
    for (Py_ssize_t index = 0; index < PyList_GET_SIZE(keys); index++) {
        if (insert(self, PyList_GET_ITEM(self->keys, index), index, -1)) {
            Py_DECREF(self);
            return NULL;
        }
    }
    return (PyObject *)self;
}


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


static PyMethodDef fam_methods[] = {
    {"__getnewargs__", (PyCFunction) fam___getnewargs__, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction) fam___reversed__, METH_NOARGS, NULL},
    {"__sizeof__", (PyCFunction) fam___sizeof__, METH_NOARGS, NULL},
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
