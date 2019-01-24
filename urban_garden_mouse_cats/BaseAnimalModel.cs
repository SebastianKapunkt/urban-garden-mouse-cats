using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace catandmouse
{
    public class BaseAnimalModel
    {
        public Variable<double> Population;
        public Variable<Gaussian> BornYoungPerLitterPrior;
        public Variable<Gaussian> BirthratePrior;
        public Variable<Gaussian> DeathratePrior;

        public Variable<double> BornYoungPerLitter;
        public Variable<double> Birthrate;
        public Variable<double> Deathrate;

        public virtual void CreateModel()
        {
            BornYoungPerLitterPrior = Variable.New<Gaussian>();
            BirthratePrior = Variable.New<Gaussian>();
            DeathratePrior = Variable.New<Gaussian>();
            Population = Variable.New<double>();

            BornYoungPerLitter = Variable.Random<double, Gaussian>(BornYoungPerLitterPrior);
            Birthrate = Variable.Random<double, Gaussian>(BirthratePrior);
            Deathrate = Variable.Random<double, Gaussian>(DeathratePrior);
        }

        public virtual void SetModelData(AnimalModelData priors)
        {
            BornYoungPerLitterPrior.ObservedValue = priors.BornYoungPerLitterDist;
            BirthratePrior.ObservedValue = priors.BirthrateDist;
            DeathratePrior.ObservedValue = priors.DeathrateDist;
        }

        public void SetNewPopulation(double NewPopulation){
            Population.ObservedValue = NewPopulation;
        }

        public Variable<double> GetNaturalDeath(){
            return Deathrate * Population;
        }

        public Variable<double> GetFemine(){
            return Population * 0.5;
        }

        public Variable<double> GetBornYoung(){
            return BornYoungPerLitter * Birthrate * GetFemine();
        }

        public struct AnimalModelData
        {
            public Gaussian BornYoungPerLitterDist;
            public Gaussian BirthrateDist;
            public Gaussian DeathrateDist;

            public AnimalModelData(
                Gaussian BornYoungPerLitterDist,
                Gaussian BirthrateDist,
                Gaussian DeathrateDist
            )
            {
                this.BornYoungPerLitterDist = BornYoungPerLitterDist;
                this.BirthrateDist = BirthrateDist;
                this.DeathrateDist = DeathrateDist;
            }
        }
    }
}