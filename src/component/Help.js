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
                <h1 style  = {{fontSize: 30, fontFamily: "Roboto", color: "#247AFC"}}>Membership Inference Attack (MIA)
                    <IconButton sx = {{marginLeft: 90, marginTop: -2}} onClick = {() => props.setPopup(false)}>
                        <CloseOutlinedIcon/>
                    </IconButton>
                    <hr />
                </h1>
                <div style = {{fontSize: 25, fontFamile: "Roboto", color: "black", fontWeight: 10, textAlign: "left", marginLeft: 5}}>
                {(lan == 'Eng')
                ?
                <p>
                Membership Inference Attack (MIA) is a method used to check if an AI model has been trained on a particular data point. This method is based on the idea that an AI model usually predicts better on data it has already been trained on. 
                If your data has been trained, the resulting MIA score will be high. If your data has not been trained, its MIA score will be low.
                </p>
                :
                <p>
                    MIA는 AI 모델이 특정 데이터 포인트에 대해 학습되었는지 확인하는 데 사용되는 방법입니다. 이 방법은 일반적으로 AI 모델이 학습에 사용한 데이터에 대해 더 잘 예측한다는 아이디어에 기반을 두고 있습니다. 따라서 데이터가 학습에 사용되었을 경우, MIA 점수가 높을 것입니다. 반대로 데이터가 학습에 사용되지 않은 경우 MIA 점수는 낮습니다.
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
