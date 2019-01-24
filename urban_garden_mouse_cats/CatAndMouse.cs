using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace catandmouse
{

    public class CatAndMouse
    {
        public BaseAnimalModel Mouse;
        public CatModel Cat;
        public InferenceEngine engine;

        public virtual void CreateModel()
        {
            Mouse = new BaseAnimalModel();
            Cat = new CatModel();

            Mouse.CreateModel();
            Cat.CreateModel();
        }

        public void SetModelData(
            BaseAnimalModel.AnimalModelData MousePriors,
            BaseAnimalModel.AnimalModelData CatPriors,
            Gaussian CatchrateDist
        )
        {
            Mouse.SetModelData(MousePriors);
            Cat.SetModelData(CatPriors, CatchrateDist);
        }

        public Variable<double> GetCatchedMouse()
        {
            Variable<double> CatchedMouse = Variable.New<double>();

            Variable<double> CatchableMouse = Cat.GetCatchableMouse();
            Variable<double> MousePopulation = Mouse.Population;

            Variable<bool> condition = CatchableMouse > MousePopulation;
            using (Variable.If(condition))
            {
                CatchedMouse.SetTo(Mouse.Population);
            }
            using (Variable.IfNot(condition))
            {
                CatchedMouse.SetTo(CatchableMouse);
            }

            return CatchedMouse;
        }

        public Variable<double> GetDyingMouse()
        {
            return GetCatchedMouse() + Mouse.GetNaturalDeath();
        }

        public Variable<double> GetCatPopulationChange()
        {
            return Cat.GetBornYoung() - Cat.GetNaturalDeath();
        }

        public Variable<double> GetMousePopulationChange()
        {
            return Mouse.GetBornYoung() - GetDyingMouse();
        }
    }
}