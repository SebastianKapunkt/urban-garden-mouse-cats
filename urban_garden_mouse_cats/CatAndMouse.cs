using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace catandmouse
{

    public class CatAndMouse
    {
        public BaseAnimalModel Mouse;
        public CatModel Cat;

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
            Variable<double> MousePopulation = Mouse.GetPopulation();

            Variable<bool> condition = CatchableMouse > MousePopulation;
            using (Variable.If(condition))
            {
                CatchedMouse.SetTo(Mouse.GetPopulation());
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

        public VariableArray2D<double> GetPopulationForIteration(int Iterations, double CatPopulation, double MousePopulation)
        {
            Variable<int> numTimes = Variable.Observed(Iterations);
            Range time = new Range(numTimes);
            Range cols = new Range(2); // frist is Cats and second Mice

            VariableArray2D<double> days = Variable.Array<double>(time, cols);

            using (ForEachBlock rowBlock = Variable.ForEach(time))
            {
                var day = rowBlock.Index;
                using (Variable.If(day == 0))
                {
                    Cat.SetNewPopulation(CatPopulation);
                    Mouse.SetNewPopulation(MousePopulation);
                    days[day, 0] = Cat.GetPopulation();
                    days[day, 1] = Mouse.GetPopulation();
                }
                using (Variable.If(day > 0))
                {
                    days[day, 0] = days[day - 1, 0] + GetCatPopulationChange();
                    days[day, 1] = days[day - 1, 1] + GetMousePopulationChange();
                }
            }

            return days;
        }
    }
}