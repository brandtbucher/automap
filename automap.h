# ifndef STATIC_STRUCTURES_H
# define STATIC_STRUCTURES_H


# include "Python.h"


# define LOAD 0.5
# define SCAN 8


typedef struct {
    Py_hash_t hash;
    Py_ssize_t index;
} entry;


typedef struct {
    PyObject_VAR_HEAD
    Py_ssize_t size;
    entry *entries;
    PyObject* keys;
} AutoMapObject;


static PyTypeObject FrozenAutoMapType;
static PyTypeObject AutoMapType;
static PyObject* intcache = NULL;


# endif
