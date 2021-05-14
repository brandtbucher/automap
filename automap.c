// TODO: More tests.
// TODO: Group similar functionality.
// TODO: Make copies faster.
// TODO: Check refcounts when calling into hash and comparison functions.
// TODO: Check allocation and cleanup.
// TODO: Richcompare is only ==, !=, figure out view containment, etc.
// TODO: Subinterpreter support.
// TODO: GC support.
// More comments.


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

...basically looks like this (assume the hashes are 15, 30, and 40):

Indices: [ 2, --, --, --, --, --,  1,  0]

Hashes:  [15, 30, 40, --, --]
Keys:    [ a,  b,  c, --, --]
Values:  [ 0,  1,  2  --, --]

It's pretty standard; keys, values, and cached hashes are stored in sequential
order, and their offsets are placed in the Indices table at position
HASH % TABLE_SIZE. Though it's not used here, collisions are resolved by jumping
around the table according to the following recurrence:

NEXT_INDEX = (5 * CURRENT_INDEX + 1 + (HASH >>= 5)) % TABLE_SIZE

This is good in the face of bad hash algorithms, but is sorta expensive. It's
also unable to utilize cache lines at all, since it's basically random (it's
literally based on random number generation)!

To contrast, the same table looks something like this for us:

Indices: [2,  --,  1,  0, --, --, --, --, --, --, --, --, --, --, --, --, --, --, --]
Hashes:  [40, --, 30, 15, --, --, --, --, --, --, --, --, --, --, --, --, --, --, --]

Keys:    [ a,  b,  c]

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

There are a couple of other tricks that we use, like globally caching integer
objects from value lookups and skipping the table entirely for identity
mappings, but the hardware-friendly hash table design is what really gives us
our awesome performance.

*******************************************************************************/

# define PY_SSIZE_T_CLEAN
# include "Python.h"


// Py_UNREACHABLE() isn't available in Python 3.6:

# ifndef Py_UNREACHABLE
# define Py_UNREACHABLE() Py_FatalError("https://xkcd.com/2200")
# endif


// Experimentation shows that these values work well:

# define LOAD 0.9
# define SCAN 16


typedef struct {
    Py_ssize_t index;
    Py_hash_t hash;
} entry;


typedef struct {
    PyObject_VAR_HEAD
    Py_ssize_t tablesize;
    entry *table;
    PyObject* keys;
} FAMObject;


typedef enum {
    ITEMS,
    KEYS,
    VALUES,
} Kind;


typedef struct {
    PyObject_VAR_HEAD
    FAMObject *map;
    Kind kind;
} FAMVObject;


typedef struct {
    PyObject_VAR_HEAD
    FAMObject *map;
    Kind kind;
    int reversed;
    Py_ssize_t index;
} FAMIObject;


static PyTypeObject AMType;
static PyTypeObject FAMIType;
static PyTypeObject FAMVType;
static PyTypeObject FAMType;


static PyObject *intcache = NULL;
static Py_ssize_t count = 0;


static void
fami_dealloc(FAMIObject *self)
{
    Py_DECREF(self->map);
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static FAMIObject*
fami_iter(FAMIObject *self)
{
    Py_INCREF(self);
    return self;
}


static PyObject*
fami_iternext(FAMIObject *self)
{
    Py_ssize_t index;
    if (self->reversed) {
        index = PyList_GET_SIZE(self->map->keys) - ++self->index;
        if (index < 0) {
            return NULL;
        }
    }
    else {
        index = self->index++;
    }
    if (PyList_GET_SIZE(self->map->keys) <= index) {
        return NULL;
    }
    switch (self->kind) {
        case ITEMS: {
            return PyTuple_Pack(
                2,
                PyList_GET_ITEM(self->map->keys, index),
                PyList_GET_ITEM(intcache, index)
            );
        }
        case KEYS: {
            PyObject *yield = PyList_GET_ITEM(self->map->keys, index);
            Py_INCREF(yield);
            return yield;
        }
        case VALUES: {
            PyObject *yield = PyList_GET_ITEM(intcache, index);
            Py_INCREF(yield);
            return yield;
        }
        default: {
            Py_UNREACHABLE();
        }
    }
}


static PyObject*
fami_methods___length_hint__(FAMIObject* self)
{
    return PyLong_FromSsize_t(Py_MAX(0, PyList_GET_SIZE(self->map->keys) - self->index));
}


static PyObject *iter(FAMObject*, Kind, int);


static PyObject*
fami_methods___reversed__(FAMIObject* self)
{
    return iter(self->map, self->kind, !self->reversed);
}


static PyMethodDef fami_methods[] = {
    {"__length_hint__", (PyCFunction) fami_methods___length_hint__, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction) fami_methods___reversed__, METH_NOARGS, NULL},
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


static PyObject*
iter(FAMObject* map, Kind kind, int reversed)
{
    FAMIObject* self = PyObject_New(FAMIObject, &FAMIType);
    if (!self) {
        return NULL;
    }
    Py_INCREF(map);
    self->map = map;
    self->kind = kind;
    self->reversed = reversed;
    self->index = 0;
    return (PyObject*)self;
}



# define SET_OP(name, op)                                 \
static PyObject*                                          \
name(PyObject* left, PyObject* right)                     \
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
    PyObject* result = PyNumber_InPlace##op(left, right); \
    Py_DECREF(left);                                      \
    Py_DECREF(right);                                     \
    return result;                                        \
}


SET_OP(famv_and, And)
SET_OP(famv_or, Or)
SET_OP(famv_subtract, Subtract)
SET_OP(famv_xor, Xor)


# undef SET_OP


static PyNumberMethods famv_as_number = {
    .nb_and = (binaryfunc) famv_and,
    .nb_or = (binaryfunc) famv_or,
    .nb_subtract = (binaryfunc) famv_subtract,
    .nb_xor = (binaryfunc) famv_xor,
};


static int fam_contains(FAMObject*, PyObject*);
static PyObject *famv_iter(FAMVObject*);


static int
famv_contains(FAMVObject* self, PyObject* other)
{
    if (self->kind == KEYS) {
        return fam_contains(self->map, other);
    }
    PyObject *iterator = famv_iter(self);
    if (!iterator) {
        return -1;
    }
    int result = PySequence_Contains(iterator, other);
    Py_DECREF(other);
    return result;
}


static PySequenceMethods famv_as_sequence = {
    .sq_contains = (objobjproc) famv_contains,
};


static void
famv_dealloc(FAMVObject* self)
{
    Py_DECREF(self->map);
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject*
famv_iter(FAMVObject* self)
{
    return iter(self->map, self->kind, 0);
}


static PyObject*
famv_methods___length_hint__(FAMVObject* self)
{
    return PyLong_FromSsize_t(PyList_GET_SIZE(self->map->keys));
}


static PyObject*
famv_methods___reversed__(FAMVObject* self)
{
    return iter(self->map, self->kind, 1);
}


static PyObject*
famv_methods_isdisjoint(FAMVObject* self, PyObject* other)
{
    PyObject* intersection = famv_and((PyObject*)self, other);
    if (!intersection) {
        return NULL;
    }
    Py_ssize_t result = PySet_GET_SIZE(intersection);
    Py_DECREF(intersection);
    return PyBool_FromLong(result);
}


static PyMethodDef famv_methods[] = {
    {"__length_hint__", (PyCFunction) famv_methods___length_hint__, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction) famv_methods___reversed__, METH_NOARGS, NULL},
    {"isdisjoint", (PyCFunction) famv_methods_isdisjoint, METH_O, NULL},
    {NULL},
};


static PyObject*
famv_richcompare(FAMVObject* self, PyObject* other, int op)
{
    PyObject* left = PySet_New((PyObject*)self);
    if (!left) {
        return NULL;
    }
    PyObject* right = PySet_New(other);
    if (!right) {
        Py_DECREF(left);
        return NULL;
    }
    PyObject* result = PyObject_RichCompare(left, right, op);
    Py_DECREF(left);
    Py_DECREF(right);
    return result;
}


static PyTypeObject FAMVType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_number = &famv_as_number,
    .tp_as_sequence = &famv_as_sequence,
    .tp_basicsize = sizeof(FAMVObject),
    .tp_dealloc = (destructor) famv_dealloc,
    .tp_iter = (getiterfunc) famv_iter,
    .tp_methods = famv_methods,
    .tp_name = "automap.FrozenAutoMapView",
    .tp_richcompare = (richcmpfunc) famv_richcompare,
};


static PyObject*
view(FAMObject* map, int kind)
{
    FAMVObject* self = (FAMVObject*)PyObject_New(FAMVObject, &FAMVType);
    if (!self) {
        return NULL;
    }
    self->kind = kind;
    self->map = map;
    Py_INCREF(map);
    return (PyObject*)self;
}


static Py_ssize_t
lookup_hash(FAMObject* self, PyObject* key, Py_hash_t hash)
{
    entry* table = self->table;
    Py_ssize_t mask = self->tablesize - 1;
    Py_hash_t mixin = Py_ABS(hash);
    PyObject **items = PySequence_Fast_ITEMS(self->keys);
    Py_ssize_t index = hash & mask;
    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            Py_hash_t h = table[index].hash;
            if (h == -1) {
                // Miss.
                return index;
            }
            if (h != hash) {
                // Collision.
                index++;
                continue;
            }
            PyObject *guess = items[table[index].index];
            if (guess == key) {
                // Hit.
                return index;
            }
            int result = PyObject_RichCompareBool(guess, key, Py_EQ);
            if (result < 0) {
                // Error.
                return -1;
            }
            if (result) {
                // Hit.
                return index;
            }
            index++;
        }
        index = (5 * (index - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


static Py_ssize_t
lookup(FAMObject* self, PyObject* key) {
    Py_hash_t hash = PyObject_Hash(key);
    if (hash == -1) {
        return -1;
    }
    Py_ssize_t index = lookup_hash(self, key, hash);
    if ((index < 0) || (self->table[index].hash == -1)) {
        return -1;
    }
    return self->table[index].index;
}


static int
insert(FAMObject* self, PyObject* key, Py_ssize_t offset, Py_hash_t hash)
{
    if (hash == -1) {
        hash = PyObject_Hash(key);
        if (hash == -1) {
            return -1;
        }
    }
    Py_ssize_t index = lookup_hash(self, key, hash);
    if (index < 0) {
        return -1;
    }
    if (self->table[index].hash != -1) {
        PyErr_SetObject(PyExc_ValueError, key);
        return -1;
    }
    self->table[index].index = offset;
    self->table[index].hash = hash;
    return 0;
}


static int
fill_intcache(Py_ssize_t size)
{
    PyObject* item;
    for (Py_ssize_t index = PyList_GET_SIZE(intcache); index < size; index++) {
        item = PyLong_FromSsize_t(index);
        if (!item) {
            return -1;
        }
        if (PyList_Append(intcache, item)) {
            Py_DECREF(item);
            return -1;
        }
        Py_DECREF(item);
    }
    return 0;
}


static int
grow(FAMObject* self, Py_ssize_t needed)
{
    if (fill_intcache(needed)) {
        return -1;
    }
    Py_ssize_t oldsize = self->tablesize;
    Py_ssize_t newsize = 1;
    needed /= LOAD;
    while (newsize <= needed) {
        newsize <<= 1;
    }
    if (newsize <= oldsize) {
        return 0;
    }
    entry* oldentries = self->table;
    entry* newentries = PyMem_New(entry, newsize + SCAN - 1);
    if (!newentries) {
        return -1;
    }
    Py_ssize_t index;
    for (index = 0; index < newsize + SCAN - 1; index++) {
        newentries[index].hash = -1;
        newentries[index].index = -1;
    }
    self->table = newentries;
    self->tablesize = newsize;
    if (oldsize) {
        for (index = 0; index < oldsize + SCAN - 1; index++) {
            if ((oldentries[index].hash != -1) &&
                insert(self, PyList_GET_ITEM(self->keys,
                                                      oldentries[index].index),
                       oldentries[index].index, oldentries[index].hash))
            {
                PyMem_Del(self->table);
                self->table = oldentries;
                self->tablesize = oldsize;
                return -1;
            }
        }
    }
    PyMem_Del(oldentries);
    return 0;
}


static FAMObject*
new(PyTypeObject* cls, PyObject* keys)
{
    if (keys) {
        keys = PySequence_List(keys);
    }
    else {
        keys = PyList_New(0);
    }
    if (!keys) {
        return NULL;
    }
    FAMObject* self = (FAMObject*)cls->tp_alloc(cls, 0);
    if (!self) {
        Py_DECREF(keys);
        return NULL;
    }
    self->keys = keys;
    count += PyList_GET_SIZE(keys);
    if (grow(self, PyList_GET_SIZE(keys))) {
        Py_DECREF(self);
        return NULL;
    }
    for (Py_ssize_t index = 0; index < PyList_GET_SIZE(keys); index++)
    {
        if (insert(self, PyList_GET_ITEM(self->keys, index), index, -1))
        {
            Py_DECREF(self);
            return NULL;
        }
    }
    return self;
}


static int
extend(FAMObject* self, PyObject* keys)
{
    keys = PySequence_Fast(keys, "expected an iterable of keys");
    if (!keys) {
        return -1;
    }
    Py_ssize_t extendsize = PyList_GET_SIZE(keys);
    count += extendsize;
    if (grow(self, PyList_GET_SIZE(self->keys) + extendsize)) {
        Py_DECREF(keys);
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(keys);
    for (Py_ssize_t index = 0; index < extendsize; index++) {
        if (insert(self, items[index], PyList_GET_SIZE(self->keys), -1) ||
            PyList_Append(self->keys, items[index]))
        {
            Py_DECREF(keys);
            return -1;
        }
    }
    Py_DECREF(keys);
    return 0;
}


static int
append(FAMObject* self, PyObject* key)
{
    count++;
    if (grow(self, PyList_GET_SIZE(self->keys) + 1)) {
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
fam_length(FAMObject* self)
{
    return PyList_GET_SIZE(self->keys);
}


static PyObject*
get(FAMObject* self, PyObject* key, PyObject* missing) {
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
    PyObject *index = PyList_GET_ITEM(intcache, result);
    Py_INCREF(index);
    return index;
}


static PyObject*
fam_subscript(FAMObject* self, PyObject* key)
{
    return get(self, key, NULL);
}


static PyMappingMethods fam_as_mapping = {
    .mp_length = (lenfunc) fam_length,
    .mp_subscript = (binaryfunc) fam_subscript,
};


static PyObject*
fam_or(PyObject* left, PyObject* right)
{
    if (!PyObject_TypeCheck(left, &FAMType) ||
        !PyObject_TypeCheck(right, &FAMType)
    ) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    FAMObject *updated = new(Py_TYPE(left), left);
    if (!updated) {
        return NULL;
    }
    if (extend(updated, right)) {
        Py_DECREF(updated);
        return NULL;
    }
    return (PyObject*)updated;
}


static PyNumberMethods fam_as_number = {
    .nb_or = (binaryfunc) fam_or,
};


static int
fam_contains(FAMObject* self, PyObject* key)
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
fam_dealloc(FAMObject* self)
{
    PyMem_Del(self->table);
    Py_XDECREF(self->keys);
    count -= PyList_GET_SIZE(self->keys);
    if (count < PyList_GET_SIZE(intcache)) {
        // del intcache[count:]
        PyList_SetSlice(intcache, count, PyList_GET_SIZE(intcache), NULL);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static Py_hash_t
fam_hash(FAMObject* self)
{
    Py_hash_t hash = 0;
    for (Py_ssize_t i = 0; i < self->tablesize; i++) {
        hash = hash * 3 + self->table[i].hash;
    }
    if (hash == -1) {
        return 0;
    }
    return hash;
}


static PyObject*
fam_iter(FAMObject* self)
{
    return iter(self, KEYS, 0);
}


static PyObject*
fam_methods___getnewargs__(FAMObject* self)
{
    return PyTuple_Pack(1, self->keys);
}


static PyObject*
fam_methods___reversed__(FAMObject* self)
{
    return iter(self, KEYS, 1);
}


static PyObject*
fam_methods___sizeof__(FAMObject* self)
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
        + (self->tablesize + SCAN - 1) * sizeof(entry)
    );
}


static PyObject*
fam_methods_get(FAMObject* self, PyObject* args)
{
    PyObject *key, *missing = Py_None;
    if (!PyArg_UnpackTuple(args, Py_TYPE(self)->tp_name, 1, 2, &key, &missing)) {
        return NULL;
    }
    return get(self, key, missing);
}


static PyObject*
fam_methods_items(FAMObject* self)
{
    return view(self, ITEMS);
}


static PyObject*
fam_methods_keys(FAMObject* self)
{
    return view(self, KEYS);
}


static PyObject*
fam_methods_values(FAMObject* self)
{
    return view(self, VALUES);
}


static PyMethodDef fam_methods[] = {
    {"__getnewargs__", (PyCFunction) fam_methods___getnewargs__, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction) fam_methods___reversed__, METH_NOARGS, NULL},
    {"__sizeof__", (PyCFunction) fam_methods___sizeof__, METH_NOARGS, NULL},
    {"get", (PyCFunction) fam_methods_get, METH_VARARGS, NULL},
    {"items", (PyCFunction) fam_methods_items, METH_NOARGS, NULL},
    {"keys", (PyCFunction) fam_methods_keys, METH_NOARGS, NULL},
    {"values", (PyCFunction) fam_methods_values, METH_NOARGS, NULL},
    {NULL},
};


static PyObject*
fam_new(PyTypeObject* cls, PyObject* args, PyObject* kwargs)
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
    return (PyObject*)new(cls, keys);
}


static PyObject*
fam_repr(FAMObject* self)
{
    return PyUnicode_FromFormat("%s(%R)", Py_TYPE(self)->tp_name, self->keys);
}


static PyObject*
fam_richcompare(FAMObject* self, PyObject* other, int op)
{
    // TODO
    if (!PyObject_TypeCheck(other, &FAMType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    PyObject *other_keys = ((FAMObject*)other)->keys;
    if ((PyObject*)self == other || self->keys == other_keys) {
        return PyBool_FromLong(op == Py_EQ || op == Py_GE || op == Py_LE);
    }
    if (Py_TYPE(self->keys) == Py_TYPE(other_keys)) {
        return PyObject_RichCompare(self->keys, other_keys, op);
    }
    Py_ssize_t len = PyList_GET_SIZE(self->keys);
    Py_ssize_t other_len = PyList_GET_SIZE(other_keys);
    Py_ssize_t common = Py_MIN(len, other_len);
    for (Py_ssize_t i = 0; i < common; i++) {
        int result = PyObject_RichCompareBool(
                         PyList_GET_ITEM(self->keys, i),
                         PyList_GET_ITEM(other_keys, i),
                         op);
        if (result < 0) {
            return NULL;
        }
        if (!result) {
            Py_RETURN_FALSE;
        }
    }
    switch (op) {
        case Py_EQ:
            return PyBool_FromLong(len == other_len);
        case Py_GE:
            return PyBool_FromLong(len >= other_len);
        case Py_GT:
            return PyBool_FromLong(len > other_len);
        case Py_LE:
            return PyBool_FromLong(len <= other_len);
        case Py_LT:
            return PyBool_FromLong(len < other_len);
        case Py_NE:
            return PyBool_FromLong(len != other_len);
        default:
            Py_UNREACHABLE();
    }
}


static PyTypeObject FAMType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_mapping = &fam_as_mapping,
    .tp_as_number = &fam_as_number,
    .tp_as_sequence = &fam_as_sequence,
    .tp_basicsize = sizeof(FAMObject),
    .tp_dealloc = (destructor) fam_dealloc,
    .tp_doc = "An immutable autoincremented integer-valued mapping.",
    .tp_hash = (hashfunc) fam_hash,
    .tp_iter = (getiterfunc) fam_iter,
    .tp_methods = fam_methods,
    .tp_name = "automap.FrozenAutoMap",
    .tp_new = fam_new,
    .tp_repr = (reprfunc) fam_repr,
    .tp_richcompare = (richcmpfunc) fam_richcompare,
};


static PyObject*
am_inplace_or(FAMObject* self, PyObject* other)
{
    if (extend(self, other)) {
        return NULL;
    }
    Py_INCREF(self);
    return (PyObject*)self;
}


static PyNumberMethods am_as_number = {
    .nb_inplace_or = (binaryfunc) am_inplace_or,
};


static PyObject*
am_methods_add(FAMObject* self, PyObject* other)
{
    if (append(self, other)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject*
am_methods_update(FAMObject* self, PyObject* other)
{
    if (extend(self, other)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyMethodDef am_methods[] = {
    {"add", (PyCFunction) am_methods_add, METH_O, NULL},
    {"update", (PyCFunction) am_methods_update, METH_O, NULL},
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


/* automap **********************************************************/


void
automap_free(PyObject* self)
{
    Py_CLEAR(intcache);
};


static struct PyModuleDef automap_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_doc = "High-performance autoincremented integer-valued mappings.",
    .m_free = (freefunc) automap_free,
    .m_name = "automap",
    .m_size = sizeof(PyObject*),
};


PyObject*
PyInit_automap(void)
{
    PyObject* automap = PyModule_Create(&automap_module);
    if (
        !automap
        || PyType_Ready(&AMType)
        || PyType_Ready(&FAMIType)
        || PyType_Ready(&FAMVType)
        || PyType_Ready(&FAMType)
        || PyModule_AddObject(automap, "AutoMap", (PyObject*)&AMType)
        || PyModule_AddObject(automap, "FrozenAutoMap", (PyObject*)&FAMType)
        || !(intcache = PyList_New(0))
    ) {
        Py_XDECREF(automap);
        return NULL;
    }
    return automap;
}
