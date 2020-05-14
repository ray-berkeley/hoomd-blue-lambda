// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause Licence


// Maintainer: gld215

#ifndef __PAIR_EVALUATOR_LJ_LAMBDA_H__
#define __PAIR_EVALUATOR_LJ_LAMBDA_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairLJLambda.h
    \brief Defines the pair evaluator class for LJ perturbation potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

//! Lennard-Jones lambda parameters
/*!
 * See EvaluatorPairLJLambda for details of the parameters.
 */
struct lj_lambda_params
    {
    Scalar lj1; //<! The coefficient for 1/r^12
    Scalar lj2; //!< The coefficient for 1/r^6
    Scalar lam; //!< Controls the attractive tail, between 0 and 1
    Scalar rwcasq; //!< The square of the location of the LJ potential minimum
    Scalar wca_shift; //!< The amount to shift the repulsive part by
    };

//! Convenience function for making a lj_lambda_params in python
HOSTDEVICE inline lj_lambda_params make_lj_lambda_params(Scalar lj1, Scalar lj2, Scalar lam, Scalar rwcasq, Scalar wca_shift)
    {
    lj_lambda_params p;
    p.lj1 = lj1;
    p.lj2 = lj2;
    p.lam = lam;
    p.rwcasq = rwcasq;
    p.wca_shift = wca_shift;
    return p;
    }

//! Class for evaluating the LJ-lambda pair potential
/*!
    EvaluatorPairLJLambda evaluates the function:
        \f{eqnarray*}{
        V(r)  = & V_{\mathrm{LJ}}(r, \varepsilon, \sigma) + (1-\lambda)\varepsilon & r < 2^{1/6}\sigma \\
              = & \lambda V_{\mathrm{LJ}}(r, \varepsilon, \sigma) & 2^{1/6}\sigma \ge r < r_{\mathrm{cut}} \\
              = & 0 & r \ge r_{\mathrm{cut}}
        \f}

    where \f$V_{\mathrm{LJ}}(r,\varepsilon,sigma)\f$ is the standard Lennard-Jones potential (see EvaluatorPairLJ)
    with parameters \f$\varepsilon\f$, \f$\sigma\f$, and \f$\alpha=1\f$.

    The LJ potential does not need diameter or charge. Five parameters are specified and stored in a
    lj_lambda_params. These are related to the standard lj parameters sigma and epsilon by:
    - \a lj1 = 4.0 * epsilon * pow(sigma,12.0)
    - \a lj2 = alpha * 4.0 * epsilon * pow(sigma,6.0);
    - \a lambda ranges from 0 to 1 and sets how close the potential is to WCA or LJ (0 = WCA, 1 = LJ)
    - \a rwcasq is the square of the location of the potential minimum (WCA cutoff), pow(2.0,1./3.) * sigma * sigma
    - \a wca_shift is the amount needed to shift the energy of the repulsive part to match the attractive energy.

*/
class EvaluatorPairLJLambda
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef lj_lambda_params param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _rcutsq Sqauared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairLJLambda(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), lj1(_params.lj1), lj2(_params.lj2), lam(_params.lam),
              rwcasq(_params.rwcasq), wca_shift(_params.wca_shift)
            {
            }

        //! LJ lambda doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! LJ lambda doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff

            \return True if they are evaluated or false if they are not because we are beyond the cuttoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            if (rsq < rcutsq && lj1 != 0)
                {
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar r6inv = r2inv * r2inv * r2inv;
                force_divr= r2inv * r6inv * (Scalar(12.0)*lj1*r6inv - Scalar(6.0)*lj2);

                pair_eng = r6inv * (lj1*r6inv - lj2);
                if (rsq < rwcasq)
                    {
                    pair_eng += wca_shift;
                    }
                else
                    {
                    force_divr *= lam;
                    pair_eng *= lam;
                    }

                if (energy_shift)
                    {
                    Scalar rcut2inv = Scalar(1.0)/rcutsq;
                    Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                    pair_eng -= lam * rcut6inv * (lj1*rcut6inv - lj2);
                    }
                return true;
                }
            else
                return false;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("lj_lambda");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar lj1;     //!< lj1 parameter extracted from the params passed to the constructor
        Scalar lj2;     //!< lj2 parameter extracted from the params passed to the constructor
        Scalar lam;  //!< lambda parameter
        Scalar rwcasq;  //!< WCA cutoff radius squared
        Scalar wca_shift; //!< Energy shift for WCA part of the potential
    };

#undef DEVICE
#undef HOSTDEVICE

#endif // __PAIR_EVALUATOR_LJ_H__
