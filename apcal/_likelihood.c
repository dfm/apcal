#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>

/* Docstrings */
static char doc[] =
"Faster likelihood calculation for an *awesome* calibration model\n";

static char lnlikelihood_doc[] =
"Calculate the ln-likelihood\n\n"\
"Parameters\n"\
"----------\n"\
"model : PatchModel\n"\
"    The model object.\n";

PyMODINIT_FUNC init_likelihood(void);
static PyObject *likelihood_lnlikelihood(PyObject *self, PyObject *args);
static PyObject *likelihood_lnoddsvar(PyObject *self, PyObject *args);
static PyObject *likelihood_lnoddsbad(PyObject *self, PyObject *args);
static PyObject *likelihood_lnlikeratiobad(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"lnlikelihood",   likelihood_lnlikelihood,   METH_VARARGS, lnlikelihood_doc},
    {"lnoddsvar",      likelihood_lnoddsvar,      METH_VARARGS, "Calculate the odds that a star is variable."},
    {"lnoddsbad",      likelihood_lnoddsbad,      METH_VARARGS, "Calculate the odds that a specific measurement is bad."},
    {"lnlikeratiobad", likelihood_lnlikeratiobad, METH_VARARGS, "Calculate the likelihood ratio that a specific measurement is bad."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_likelihood(void)
{
    PyObject *m = Py_InitModule3("_likelihood", module_methods, doc);
    if (m == NULL)
        return;

    import_array();
}

/* ================================= //
// PYTHON DATA TYPE HELPER FUNCTIONS //
// ================================= */

double _fromPyDouble(PyObject *obj, char *attr, int *info)
{
    PyObject *number = PyObject_GetAttrString(obj,attr);
    if (number == NULL && *info == 0) {
        *info = 1;
        PyErr_SetString(PyExc_AttributeError, attr);
        return 0;
    }
    double result = PyFloat_AsDouble(number);
    Py_DECREF(number);
    return result;
}
PyObject *_fromNPYArray(PyObject *obj, char *attr)
{
    PyObject *arr = PyObject_GetAttrString(obj,attr);
    if (arr == NULL) {
        Py_XDECREF(arr);
        PyErr_SetString(PyExc_AttributeError, attr);
        return NULL;
    }

    PyObject *result = PyArray_FROM_OTF(arr, NPY_DOUBLE, NPY_IN_ARRAY);

    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "couldn't access numpy.ndarray object");
    }

    Py_DECREF(arr);
    return result;
}

/* ===================== //
// PURE-C MODEL "OBJECT" //
// ===================== */

typedef struct patchmodel {
    PyObject *df, *di;
    double *flux, *ivar;
    int nstars, nobs;
    PyObject *mz, *mf;
    double *zero, *fstar;
    double jitterabs2, jitterrel2, Q2, Pvar, sigbad2, Pbad;
} PatchModel;

PatchModel *PatchModel_init(PyObject *model)
{
    PatchModel *self = malloc(sizeof(PatchModel));

    // parse the data/model classes
    self->df = _fromNPYArray(model,"_f");
    self->di = _fromNPYArray(model,"_ivar_f");
    if (self->df == NULL || self->di == NULL)
        return NULL;

    self->flux = (double *)PyArray_DATA(self->df);
    self->ivar = (double *)PyArray_DATA(self->di);
    self->nobs = PyArray_DIMS(self->df)[1];
    self->nstars = PyArray_DIMS(self->df)[0];

    // parse the data/model classes
    self->mz = _fromNPYArray(model,"_f0");
    self->mf = _fromNPYArray(model,"_fstar");
    if (self->mz == NULL || self->mf == NULL)
        return NULL;

    self->zero = (double *)PyArray_DATA(self->mz);
    self->fstar = (double *)PyArray_DATA(self->mf);

    // model parameters
    int info = 0;
    self->jitterabs2 = _fromPyDouble(model,"_jitterabs2", &info);
    self->jitterrel2 = _fromPyDouble(model,"_jitterrel2", &info);
    self->Q2         = _fromPyDouble(model,"_Q2", &info);
    self->Pvar       = _fromPyDouble(model,"_pvar", &info);
    self->sigbad2    = _fromPyDouble(model,"_sigbad2", &info);
    self->Pbad       = _fromPyDouble(model,"_pbad", &info);
    if (info == 1)
        return NULL;

    return self;
}

void PatchModel_destroy(PatchModel *self)
{
    if (self != NULL) {
        Py_XDECREF(self->df);
        Py_XDECREF(self->di);

        Py_XDECREF(self->mz);
        Py_XDECREF(self->mf);

        free(self);
    }
}

// ============== //
// MATH FUNCTIONS //
// ============== //

static const double lisqrt2pi = -0.91893853320467267; // -0.5*log(2.0*M_PI);
double _lnnormal(double x, double mu, double var)
{
    double diff = x-mu;
    return -0.5*diff*diff/var - 0.5*log(var) + lisqrt2pi;
}
double _dlnnormaldmu(double x, double mu, double var)
{
    return (x-mu)/var;
}
double _dlnnormaldvar(double x, double mu, double var)
{
    double diff = x-mu;
    return 0.5*(diff*diff-var)/var;
}

double _logsumexp(double a, double b)
{
    if (a > b)
        return a + log(1+exp(b-a));
    return b + log(1+exp(a-b));
}

// ======================= //
// LIKELIHOOD CALCULATIONS //
// ======================= //

void get_lnpgood_and_lnpbad_and_lnpvargood(int i, int alpha,
        double *lnpgood, double *lnpbad, double *lnpvargood, double *lnpvarbad,
        PatchModel *model)
{
    int ind = alpha*model->nobs + i;
    *lnpgood = 0.0; *lnpbad = 0.0; *lnpvargood = 0.0;

    if (model->ivar[ind] > 0.0) {
        double ff = model->zero[i]*model->fstar[alpha];
        double ffabs = fabs(ff*ff);
        double sig2   = 1.0/model->ivar[ind];
        double delta2 = model->jitterabs2 + model->jitterrel2*ffabs;
        double sigvar2 = model->Q2; //*ffabs;
        /* printf("%e %e\n", sigvar2, model->sigbad2); */

        *lnpgood    = _lnnormal(model->flux[ind],ff,sig2+delta2);
        *lnpbad     = _lnnormal(model->flux[ind],ff,sig2+delta2+model->sigbad2);
        *lnpvargood = _lnnormal(model->flux[ind],ff,sig2+delta2+sigvar2);
        *lnpvarbad  = _lnnormal(model->flux[ind],ff,sig2+delta2+sigvar2+model->sigbad2);
    }
}

void get_lnpvar_and_lnpconst(int alpha, double *lnpvar, double *lnpconst,
        PatchModel *model)
{
    int i;
    *lnpconst = 0.0;
    *lnpvar   = 0.0;
    for (i = 0; i < model->nobs; i++) {
        double lnpgood,lnpbad,lnpvargood,lnpvarbad;
        get_lnpgood_and_lnpbad_and_lnpvargood(i,alpha,
                &lnpgood,&lnpbad,&lnpvargood,&lnpvarbad,
                model);
        *lnpconst += _logsumexp(log(1-model->Pbad)+lnpgood,
                                 log(model->Pbad)+lnpbad);
        *lnpvar   += _logsumexp(log(1-model->Pbad)+lnpvargood,
                                 log(model->Pbad)+lnpvarbad);
    }
}

double lnlikelihood(PatchModel *model)
{
    int alpha;
    double lnlike = 0.0;
#pragma omp parallel for default(shared) private(alpha) schedule(static) reduction(+:lnlike)
    for (alpha = 0; alpha < model->nstars; alpha++) {
        double lnpconst, lnpvar;
        get_lnpvar_and_lnpconst(alpha, &lnpvar, &lnpconst, model);
        lnlike += _logsumexp(log(1-model->Pvar)+lnpconst,log(model->Pvar)+lnpvar);
    }
    return lnlike;
}

void get_lnoddsvar(PatchModel *model, double *lnoddsvar)
{
    int alpha;
#pragma omp parallel for default(shared) private(alpha) schedule(static)
    for (alpha = 0; alpha < model->nstars; alpha++) {
        double lnpconst, lnpvar;

        get_lnpvar_and_lnpconst(alpha, &lnpvar, &lnpconst, model);

        lnoddsvar[alpha] = log(model->Pvar)-log(1-model->Pvar)
            + lnpvar - lnpconst;
    }
}

void get_lnlikeratiobad(PatchModel *model, double *lnlikeratiobad)
{
    int i,alpha;
#pragma omp parallel for default(shared) private(alpha) schedule(static)
    for (i = 0; i < model->nobs; i++) {
        for (alpha = 0; alpha < model->nstars; alpha++) {
            int ind = i*model->nstars+alpha;

            double lnpgood,lnpbad,lnpvargood, lnpvarbad;
            double lnpconst,lnpvar,lntotgood;

            get_lnpgood_and_lnpbad_and_lnpvargood(i,alpha,
                    &lnpgood,&lnpbad,&lnpvargood, &lnpvarbad, model);
            get_lnpvar_and_lnpconst(alpha, &lnpvar, &lnpconst, model);

            //lntotgood = _logsumexp(lnpvar+lnpvargood,lnpconst+lnpgood);
            lntotgood = lnpgood;
            lnlikeratiobad[ind] = lnpbad - lntotgood;
        }
    }
}

void get_lnoddsbad(PatchModel *model, double *lnoddsbad)
{
    int i,alpha;
#pragma omp parallel for default(shared) private(i) schedule(static)
    for (alpha = 0; alpha < model->nstars; alpha++) {
        for (i = 0; i < model->nobs; i++) {
            int ind = alpha*model->nobs + i;

            double lnpgood,lnpbad,lnpvargood, lnpvarbad;
            double lnpconst,lnpvar,lntotgood, lntotbad;

            get_lnpgood_and_lnpbad_and_lnpvargood(i,alpha,
                    &lnpgood,&lnpbad,&lnpvargood, &lnpvarbad, model);
            get_lnpvar_and_lnpconst(alpha, &lnpvar, &lnpconst, model);

            lntotgood = _logsumexp(log(model->Pvar)+lnpvar+lnpvargood,
                                lnpconst+lnpgood+log(1-model->Pvar));
            lntotbad = _logsumexp(log(model->Pvar)+lnpvar+lnpvarbad,
                                lnpconst+lnpbad+log(1-model->Pvar));

            lnoddsbad[ind] = log(model->Pbad)-log(1-model->Pbad)
                + lntotbad - lntotgood;
        }
    }
}

// ============== //
// MODULE METHODS //
// ============== //

static PyObject *likelihood_lnlikelihood(PyObject *self, PyObject *args)
{
    PyObject *model0 = NULL;
    PatchModel *model = NULL;
    if (!PyArg_ParseTuple(args, "O", &model0))
        return NULL;

    model = PatchModel_init(model0);
    if (model == NULL) {
        PatchModel_destroy(model);
        return NULL;
    }

    // Calcuate the likelihood
    double lnlike = lnlikelihood(model);
    PyObject *result = PyFloat_FromDouble(lnlike);

    // clean up!
    PatchModel_destroy(model);
    return result;
}

static PyObject *likelihood_lnoddsvar(PyObject *self, PyObject *args)
{
    PyObject *model0 = NULL, *lnoddsvar_obj = NULL;
    PyObject *lnoddsvar = NULL;
    PatchModel *model = NULL;
    if (!PyArg_ParseTuple(args, "OO", &model0, &lnoddsvar_obj))
        return NULL;

    model = PatchModel_init(model0);
    if (model == NULL) {
        PatchModel_destroy(model);
        return NULL;
    }

    lnoddsvar = PyArray_FROM_OTF(lnoddsvar_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    if (lnoddsvar == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "couldn't access numpy.ndarray object");

        Py_XDECREF(lnoddsvar);
        PatchModel_destroy(model);
        return NULL;
    }

    get_lnoddsvar(model, (double *)PyArray_DATA(lnoddsvar));

    // clean up!
    Py_DECREF(lnoddsvar);
    PatchModel_destroy(model);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *likelihood_lnoddsbad(PyObject *self, PyObject *args)
{
    PyObject *model0 = NULL, *lnoddsbad_obj = NULL, *lnoddsbad = NULL;
    PatchModel *model = NULL;
    if (!PyArg_ParseTuple(args, "OO", &model0, &lnoddsbad_obj))
        return NULL;

    model = PatchModel_init(model0);
    if (model == NULL) {
        PatchModel_destroy(model);
        return NULL;
    }

    lnoddsbad = PyArray_FROM_OTF(lnoddsbad_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    if (lnoddsbad == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "couldn't access numpy.ndarray object");

        Py_XDECREF(lnoddsbad);
        PatchModel_destroy(model);
        return NULL;
    }
    get_lnoddsbad(model, (double *)PyArray_DATA(lnoddsbad));

    // clean up!
    Py_DECREF(lnoddsbad);
    PatchModel_destroy(model);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *likelihood_lnlikeratiobad(PyObject *self, PyObject *args)
{
    PyObject *model0 = NULL, *lnlikeratiobad_obj = NULL, *lnlikeratiobad = NULL;
    PatchModel *model = NULL;
    if (!PyArg_ParseTuple(args, "OO", &model0, &lnlikeratiobad_obj))
        return NULL;

    model = PatchModel_init(model0);
    if (model == NULL) {
        PatchModel_destroy(model);
        return NULL;
    }

    lnlikeratiobad = PyArray_FROM_OTF(lnlikeratiobad_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    if (lnlikeratiobad == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "couldn't access numpy.ndarray object");

        Py_XDECREF(lnlikeratiobad);
        PatchModel_destroy(model);

        return NULL;
    }
    get_lnlikeratiobad(model, (double *)PyArray_DATA(lnlikeratiobad));

    // clean up!
    Py_DECREF(lnlikeratiobad);
    PatchModel_destroy(model);
    Py_INCREF(Py_None);
    return Py_None;
}

