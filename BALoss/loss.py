def DAHU_loss(inputs, seed, labels):
    
    # inputs:  edge probability map
    # seed: initializated seed for inside and outside regions 
    # labels: boundary for each region 


    EPM = inputs.squeeze()
    labels = labels.squeeze()

    seed = seed.squeeze(axis = 0).squeeze(axis = 0)

    d, _, _ = seed.shape
    loss_total = 0
    for i in range(d):
        loss = 0
        EPM_s, seed_s = EPM, seed[i,:,:]

        max_range = torch.max(seed_s).type(torch.uint8)
        seed_s = np.array(seed_s.detach().cpu().numpy()).astype(np.uint8)

        EPM_s = torch.sigmoid(EPM_s)

        EPM_s_clone = EPM_s.clone()
        EPM_s_clone = np.array((EPM_s_clone*255).detach().cpu().numpy()).astype(np.uint8)

        if max_range>1:

            inside = np.array(seed_s == 1).astype(np.uint8)*255
            outside = np.array(seed_s > 1).astype(np.uint8)*255
            destination = np.array(seed_s).astype(np.uint8)
            contour = MBD.MBD_cut(EPM_s_clone,inside, outside)
            kernel = np.ones((7,7),np.uint8)
            contour = cv2.dilate(contour,kernel,iterations = 1)
            contour = (contour/255).astype(np.uint8)
            contour = torch.from_numpy(np.array([contour])).cuda()
                      
            EPM_contour = (EPM_s * contour).unsqueeze(axis=0)
            label_s = (labels * contour).unsqueeze(axis=0)
            BCE_loss = cross_entropy_loss2d(EPM_contour, label_s, True, 1.1)*10
            loss = loss + BCE_loss 

        else:
            loss = torch.tensor(0).type(torch.float64)
        loss_total = loss_total + loss # Accumulate sum
    loss_total = loss_total / d
    return loss_total