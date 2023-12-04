import './style.css'
import { Fragment, useState, useEffect } from 'react'
import CloseOutlinedIcon from '@mui/icons-material/CloseOutlined';
import { Button, IconButton, Stack } from '@mui/material';


export default function Help(props){
    const [lan, setLan] = useState('Eng')
    const [helpPopup, setHelpPopup] = useState(false)
    const HelpPopupHandler = () => {
        setHelpPopup(true)
    }
    return(
        <Fragment>
        <div className='popup-shadow'>
            <div className='popup-help'>
                <h1 style  = {{fontSize: 30, fontFamily: "Roboto", color: "#247AFC"}}>Negative Gradient (NegGrad)
                    <IconButton sx = {{marginLeft: 100, marginTop: -2}} onClick = {() => props.setPopup(false)}>
                        <CloseOutlinedIcon/>
                    </IconButton>
                    <hr />
                </h1>
                <div style = {{fontSize: 25, fontFamile: "Roboto", color: "black", fontWeight: 10, textAlign: "left", marginLeft: 5}}>
                {(lan == 'Eng')
                ?
                <p>
                    Negative Gradient(NegGrad) is an algorithm used for machine unlearning. This method essentially makes the model look at the data it should forget and updates the model by making it move in the direction where the model performance gets worse for that data. 
                </p>
                :
                <p>
                    Negative Gradient (NegGrad)는 언러닝에 사용되는 알고리즘입니다. 이 방법은 모델이 잊어야 할 데이터에 대해 모델 성능이 악화되는 방향으로 움직이도록 하여 모델을 업데이트하는 방식입니다.
                </p>
                }

                </div>
                <hr />
                <Stack direction = "row" justifyContent = "center" sx = {{marginTop: 2}}>
                    <Button variant = "contained" sx = {{width: 100, height: 40}} onClick = {() => setLan('Kor')}>Kor</Button>
                    <Button variant = "outlined" sx = {{marginLeft: 1, width: 100, height: 40}} onClick = {() => setLan('Eng')}>Eng</Button>
                </Stack>
            </div>         
        </div>
        </Fragment>
    )
}
