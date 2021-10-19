#include <Python.h>
#include <assert.h>
#include "numpy/arrayobject.h"
#include "MBD_distance_2d.h"
#include <iostream>
using namespace std;


static PyObject *
MBD_cut_wrapper(PyObject *self, PyObject *args)
{
    PyObject *I=NULL, *Seed=NULL, *Destination=NULL;
    PyArrayObject *arr_I=NULL, *arr_Seed=NULL, *arr_Destination=NULL;
    
    if (!PyArg_ParseTuple(args, "OOO", &I, &Seed, &Destination)) return NULL;
    
    arr_I = (PyArrayObject*)PyArray_FROM_OTF(I, NPY_UINT8, NPY_IN_ARRAY);
    if (arr_I == NULL) return NULL;
    
    arr_Seed = (PyArrayObject*)PyArray_FROM_OTF(Seed, NPY_UINT8, NPY_IN_ARRAY);
    if (arr_Seed == NULL) return NULL;
    
    arr_Destination = (PyArrayObject*)PyArray_FROM_OTF(Destination, NPY_UINT8, NPY_IN_ARRAY);
    if (arr_Destination == NULL) return NULL;

    int nd = PyArray_NDIM(arr_I);   //number of dimensions
    npy_intp * shape = PyArray_DIMS(arr_I);  // npy_intp array of length nd showing length in each dim.
    npy_intp * shape_seed = PyArray_DIMS(arr_Seed);
    npy_intp * shape_destination = PyArray_DIMS(arr_Destination);

    // cout<<"input shape ";
    // for(int i=0; i<nd; i++)
    // {
    //     cout<<shape[i]<<" ";
    //     if(i < 2 && shape[i]!=shape_seed[i])
    //     {
    //         cout<<"input shape does not match"<<endl;
    //         return NULL;
    //     }
    // }
    // cout<<std::endl;
    // int channel = 1;
    // if(nd == 3){
    //     channel = shape[2];
    // }

    npy_intp output_shape[2];
    output_shape[0] = shape[0];
    output_shape[1] = shape[1];

    PyArrayObject * shortestpath = (PyArrayObject*)  PyArray_SimpleNew(2, output_shape, NPY_UINT8);
    MBD_cut((const unsigned char *)arr_I->data, (const unsigned char *)arr_Seed->data, 
           (const unsigned char *)arr_Destination->data, (unsigned char *) shortestpath->data, shape[0], shape[1]);
    
    Py_DECREF(arr_I);
    Py_DECREF(arr_Seed);
    Py_DECREF(arr_Destination);

    //Py_INCREF(distance);
    return PyArray_Return(shortestpath);
}

static PyMethodDef Methods[] = {
    {"MBD_cut",  MBD_cut_wrapper, METH_VARARGS, "computing 2d MBD cut"},
    // {NULL, NULL, 0, NULL}
};
