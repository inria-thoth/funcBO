import time
import torch
import wandb

class ClassicalBilevelProblem:
  """
  Instanciates the bilevel problem and solves it using Neural Implicit Differentiation.
  """

  def __init__(self, outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_model, device, batch_size=64, max_inner_iters=200, max_inner_dual_iters=5, meta_optimizer=None):
    """
    Init method.
      param outer_loss: outer level loss function
      param inner_loss: inner level loss function
      param outer_dataloader: input data and labels for the outer loss
      param inner_dataloader: input data and labels for the inner loss
      param batch_size: batch size for both inner and outer optimizers
      param device: CPU or GPU
    """
    self.__input_check__(outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_model, device, batch_size)
    self.outer_loss = outer_loss
    self.inner_loss = inner_loss
    self.outer_dataloader = outer_dataloader
    self.inner_dataloader = inner_dataloader
    self.batch_size = batch_size
    self.device = device
    self.outer_model, self.outer_optimizer, self.outer_scheduler = outer_model
    self.inner_model, self.inner_optimizer, self.inner_scheduler = inner_model
    self.max_inner_iters = max_inner_iters
    self.meta_optimizer = meta_optimizer
    self.n_meta_loss_accum = 1

  def __input_check__(self, outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_model, device, batch_size):
    """
    Ensure that the inputs are of the right form and all necessary inputs have been supplied.
    """
    if (outer_loss is None) or (inner_loss is None):
      raise AttributeError("You must specify inner and outer losses.")
    if (outer_dataloader is None) or (inner_dataloader is None):
      raise AttributeError("You must give outer and inner dataloaders.")
    if (outer_model is None) or not(len(outer_model)==3):
      raise AttributeError("You must specify the outer model, its optimizers and its scheduler.")
    if (inner_model is None) or not(len(inner_model)==3):
      raise AttributeError("You must specify the inner model, its optimizers and its scheduler.")
    if (device is None):
      raise AttributeError("You must specify a device: GPU/CPU.")
    if not (type(batch_size) is int):
      raise TypeError("Batch size must be an integer value.")

  def optimize(self, max_epochs=10, max_iters=100, eval_every_n=10):
      """
      Find the optimal outer solution.
          param maxiter: maximum number of iterations
      """
      iters, outer_losses, inner_losses, val_accs, times, evaluated, acc_smooth = 0, [], [], [], [], False, 10
      for epoch in range(max_epochs):
          for data in self.outer_dataloader:
              X_outer, main_label, aux_label, data_id = data
              aux_label = torch.stack(aux_label)
              y_outer = (main_label.to(self.device, dtype=torch.float), aux_label.to(self.device, dtype=torch.float))
              start = time.time()
              # Move data to GPU
              X_outer = X_outer.to(self.device, dtype=torch.float)
              #y_outer = y_outer.to(self.device, dtype=torch.float)
              # Inner value corresponds to h*(X_outer)
              forward_start = time.time()
              self.optimize_inner()
              inner_value = self.inner_model(X_outer)
              wandb.log({"duration of forward": time.time() - forward_start})
              loss = self.outer_loss(inner_value, y_outer)
              wandb.log({"inn. loss": self.inner_loss_val})
              wandb.log({"out. loss": loss.item()})
              wandb.log({"outer var. norm": torch.norm(self.outer_model.layer_1.weight.data).item()})
              # Backpropagation
              self.outer_optimizer.zero_grad()
              backward_start = time.time()
              self.hyperstep()
              wandb.log({"duration of backward": time.time() - backward_start})
              # Log all aux. task coeficients
              wandb.log({"outer var. norm helpful 1": torch.norm(self.outer_model.layer_1.weight.data[0,0]).item()})
              wandb.log({"outer var. norm helpful 2": torch.norm(self.outer_model.layer_1.weight.data[0,1]).item()})
              wandb.log({"outer var. norm harmful 3": torch.norm(self.outer_model.layer_1.weight.data[0,2]).item()})
              wandb.log({"outer var. norm harmful 4": torch.norm(self.outer_model.layer_1.weight.data[0,3]).item()})
              # Update loss and iteration count
              iters += 1
              duration = time.time() - start
              wandb.log({"iter. time": duration})
              times.append(duration)
              # Inner losses
              inner_losses.append(self.inner_loss_val)
              # Outer losses
              outer_losses.append(loss.item())
              if epoch_iters >= max_iters:
                  break
      return iters, outer_losses, inner_losses, val_accs, times

  def hyperstep(self):
      """
      Compute hyper-parameter gradient.
      """
      # Get loss for outer data
      iters = 0
      total_meta_val_loss = .0
      for data in self.outer_dataloader:
          X_outer, main_label, aux_label, data_id = data
          aux_label = torch.stack(aux_label)
          y_outer = (main_label.to(self.device, dtype=torch.float), aux_label.to(self.device, dtype=torch.float))
          X_outer = X_outer.to(self.device, dtype=torch.float)
          inner_value = self.inner_model(X_outer)
          loss = self.outer_loss(inner_value, y_outer)
          wandb.log({"out. loss": loss.item()})
          iters += 1
          total_meta_val_loss += loss
          meta_val_loss = total_meta_val_loss/iters
          if iters>=self.n_meta_loss_accum:
              break
      # Get loss for inner data
      iters = 0
      total_meta_train_loss = .0
      for data in self.inner_dataloader:
          X_inner, main_label, aux_label, data_id = data
          aux_label = torch.stack(aux_label)
          y_inner = (main_label.to(self.device, dtype=torch.float), aux_label.to(self.device, dtype=torch.float))
          X_inner = X_inner.to(self.device, dtype=torch.float)
          inner_value = self.inner_model(X_outer)
          loss = self.inner_loss(inner_value, y_outer)
          iters += 1
          total_meta_train_loss += loss
          meta_train_loss = total_meta_train_loss/iters
          if iters>=self.n_meta_loss_accum:
              break
      # Hyperparam step
      hypergrads = self.meta_optimizer.step(
          val_loss=meta_val_loss,
          train_loss=total_meta_train_loss,
          aux_params=list(self.outer_model.parameters()),
          parameters=list(self.inner_model.parameters()),
          return_grads=True
      )
      return hypergrads

  def optimize_inner(self):
      """
      Optimization loop for the inner-level model that approximates h*.
      """
      total_loss, total_iters = 0, 0
      for data in self.inner_dataloader:
          X_inner, main_label, aux_label, data_id = data
          aux_label = torch.stack(aux_label)
          y_inner = (main_label.to(self.device, dtype=torch.float), aux_label.to(self.device, dtype=torch.float))
          # Move data to GPU
          X_inner = X_inner.to(self.device, dtype=torch.float)
          #y_inner = y_inner.to(self.device, dtype=torch.float)
          # Get the prediction
          h_X_i = self.inner_model.forward(X_inner)
          # Compute the loss
          loss = self.inner_loss(h_X_i, y_inner)
          # Backpropagation
          self.inner_optimizer.zero_grad()
          loss.backward()
          self.inner_optimizer.step()
          total_loss += loss.item()
          total_iters += 1
          if total_iters >= self.max_inner_iters:
            break
      self.inner_loss_val = total_loss/total_iters
  
  def evaluate(self, dataloader, mu, max_iters=100):
    """
    Evaluate the prediction quality on the test dataset.
      param dataloader: data to evaluate on
      param mu: outer variable
    """
    self.inner_solution.eval = True
    total_loss, total_acc, iters = 0, 0, 0
    for data in dataloader:
      X_outer, main_label, aux_label, data_id = data
      aux_label = torch.stack(aux_label)
      y_outer = (main_label.to(self.device, dtype=torch.float), aux_label.to(self.device, dtype=torch.float))
      # Move data to GPU
      X_outer = X_outer.to(self.device, dtype=torch.float)
      # Inner value corresponds to h*(X_outer)
      inner_value = self.inner_solution(mu, X_outer, y_outer)
      # Compute loss
      loss = self.outer_loss(mu, inner_value, y_outer)
      # Compute accuracy
      _, class_pred = torch.max(inner_value[0], dim=1)
      class_pred = class_pred.to(self.device)
      accuracy = get_accuracy(class_pred, y_outer[0])
      total_acc += accuracy.item()
      total_loss += loss.item()
      iters += 1
    self.inner_solution.eval = False
    return total_loss/iters, total_acc/iters